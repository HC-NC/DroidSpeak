using DroidSpeak.Neural;

namespace DroidSpeak.Trainer;

/// <summary>
/// A dense network with mutable tensors, the backward pass of mean-squared-error training, and the
/// Adam update rule.
/// <para>
/// <see cref="MlpModel"/> is an inference object: its weights are read-only from the outside and it
/// never allocates on the hot path. This type is the training mirror. It keeps one weight, bias and
/// two Adam moment tensors per layer, allocated once in the constructor, so a full epoch afterwards
/// runs without touching the heap.
/// </para>
/// <para>
/// Weight layout matches inference exactly: row-major, <c>weights[output * inputSize + input]</c>,
/// so <see cref="Export"/> is a copy rather than a transpose, and a saved file means the same thing
/// on disk that it did in the trainer.
/// </para>
/// </summary>
internal sealed class TrainableMlp
{
    private readonly LayerSpec[] _specs;
    private readonly float[][] _weights;
    private readonly float[][] _biases;
    private readonly float[][] _weightMoments;
    private readonly float[][] _weightVelocities;
    private readonly float[][] _biasMoments;
    private readonly float[][] _biasVelocities;
    private readonly float[][] _weightGradients;
    private readonly float[][] _biasGradients;

    /// <summary>
    /// Layer outputs, one slot wider than the layer count: index 0 receives the network input and
    /// index <c>i + 1</c> holds layer <c>i</c>'s post-activation values.
    /// </summary>
    private readonly float[][] _activations;

    /// <summary>Pre-activation sums, needed by the derivative of every layer's activation.</summary>
    private readonly float[][] _preActivations;

    /// <summary>Backpropagated error signal of each layer, one value per neuron.</summary>
    private readonly float[][] _deltas;

    private readonly int _layerCount;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _epsilon;
    private int _step;

    /// <summary>Builds a trainable copy of a network shape, initialized by the inference engine.</summary>
    /// <param name="specs">Layer shapes and activations, input to output.</param>
    /// <param name="seed">
    /// Initialization seed. Borrowing <see cref="MlpModel.CreateInitialized"/> means the trainer
    /// starts from the same Glorot/He tensors a hand-built model would, so a run is reproducible from
    /// its seed alone.
    /// </param>
    /// <param name="beta1">First-moment decay.</param>
    /// <param name="beta2">Second-moment decay.</param>
    /// <param name="epsilon">Adam denominator floor.</param>
    public TrainableMlp(LayerSpec[] specs, uint seed, float beta1, float beta2, float epsilon)
    {
        _specs = specs;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _layerCount = specs.Length;

        var template = MlpModel.CreateInitialized(specs, seed);

        _weights = new float[_layerCount][];
        _biases = new float[_layerCount][];
        _weightMoments = new float[_layerCount][];
        _weightVelocities = new float[_layerCount][];
        _biasMoments = new float[_layerCount][];
        _biasVelocities = new float[_layerCount][];
        _weightGradients = new float[_layerCount][];
        _biasGradients = new float[_layerCount][];
        _preActivations = new float[_layerCount][];
        _deltas = new float[_layerCount][];
        _activations = new float[_layerCount + 1][];

        // Slot 0 is the input, slot i + 1 is layer i's output. Adjacent layers agree on the width by
        // construction (MlpModel validates the chain), so no buffer is over- or under-sized.
        _activations[0] = new float[specs[0].InputSize];

        for (var layer = 0; layer < _layerCount; layer++)
        {
            var source = template[layer];
            var width = specs[layer].InputSize;
            var height = specs[layer].OutputSize;

            _weights[layer] = source.Weights.ToArray();
            _biases[layer] = source.Biases.ToArray();
            _weightMoments[layer] = new float[width * height];
            _weightVelocities[layer] = new float[width * height];
            _biasMoments[layer] = new float[height];
            _biasVelocities[layer] = new float[height];
            _weightGradients[layer] = new float[width * height];
            _biasGradients[layer] = new float[height];
            _preActivations[layer] = new float[height];
            _deltas[layer] = new float[height];
            _activations[layer + 1] = new float[height];
        }

        InputSize = specs[0].InputSize;
        OutputSize = specs[^1].OutputSize;
    }

    /// <summary>Width of the feature vector this network consumes.</summary>
    public int InputSize { get; }

    /// <summary>Width of the target vector this network produces.</summary>
    public int OutputSize { get; }

    /// <summary>Total trainable values, reported next to the loss so a run is self-describing.</summary>
    public int ParameterCount
    {
        get
        {
            var count = 0;
            for (var layer = 0; layer < _layerCount; layer++)
            {
                count += _weights[layer].Length + _biases[layer].Length;
            }

            return count;
        }
    }

    /// <summary>Number of Adam updates applied so far (the bias-correction clock).</summary>
    public int Step => _step;

    /// <summary>
    /// Runs the forward pass, returns the mean-squared error of one sample, and accumulates its
    /// gradient into the layer buffers.
    /// </summary>
    /// <param name="input">Exactly <see cref="InputSize"/> features.</param>
    /// <param name="target">Exactly <see cref="OutputSize"/> normalized targets.</param>
    /// <returns>The sample's mean squared error.</returns>
    /// <remarks>
    /// Gradients accumulate, so the caller zeros them once per batch (<see cref="ZeroGradients"/>)
    /// instead of per sample, and the batch size is folded into the update as a reciprocal scale.
    /// </remarks>
    public float TrainSample(ReadOnlySpan<float> input, ReadOnlySpan<float> target)
    {
        Forward(input);
        return Backward(target);
    }

    /// <summary>Evaluates one sample's mean squared error without touching gradients.</summary>
    public float Evaluate(ReadOnlySpan<float> input, ReadOnlySpan<float> target)
    {
        Forward(input);
        return MeanSquaredError(target);
    }

    /// <summary>Writes the last forward pass' output into <paramref name="destination"/>.</summary>
    public void CopyOutput(Span<float> destination) => _activations[_layerCount].CopyTo(destination);

    /// <summary>Runs the forward pass and leaves the output in place for <see cref="CopyOutput"/>.</summary>
    public void Forward(ReadOnlySpan<float> input)
    {
        if (input.Length != InputSize)
        {
            throw new ArgumentException(
                $"Expected {InputSize} features, but received {input.Length}.",
                nameof(input));
        }

        input.CopyTo(_activations[0]);

        for (var layer = 0; layer < _layerCount; layer++)
        {
            var weights = _weights[layer];
            var biases = _biases[layer];
            var source = _activations[layer];
            var pre = _preActivations[layer];
            var destination = _activations[layer + 1];
            var activation = _specs[layer].Activation;
            var width = source.Length;

            for (var output = 0; output < destination.Length; output++)
            {
                var row = output * width;
                var sum = biases[output];

                for (var i = 0; i < width; i++)
                {
                    sum += weights[row + i] * source[i];
                }

                pre[output] = sum;
                destination[output] = ActivationMath.Forward(sum, activation);
            }
        }
    }

    /// <summary>Clears every accumulated gradient.</summary>
    public void ZeroGradients()
    {
        for (var layer = 0; layer < _layerCount; layer++)
        {
            Array.Clear(_weightGradients[layer]);
            Array.Clear(_biasGradients[layer]);
        }
    }

    /// <summary>
    /// Moves the weights by one Adam step and resets the accumulator.
    /// </summary>
    /// <param name="learningRate">The epoch's (possibly decayed) learning rate.</param>
    /// <param name="gradientScale">
    /// Multiplier applied to the accumulated gradient before it enters the moments: the reciprocal of
    /// the number of samples summed into it, so a batch mean behaves the same for any batch size.
    /// </param>
    /// <remarks>
    /// Bias-corrected Adam. Dividing the raw moments by <c>1 - beta^t</c> is what stops the first few
    /// steps from moving far less than later ones, which matters here because a distillation run
    /// reaches its best loss within tens of thousands of updates, not millions.
    /// </remarks>
    public void ApplyAdam(float learningRate, float gradientScale)
    {
        _step++;
        var correction1 = 1f - MathF.Pow(_beta1, _step);
        var correction2 = 1f - MathF.Pow(_beta2, _step);

        for (var layer = 0; layer < _layerCount; layer++)
        {
            Update(
                _weights[layer],
                _weightMoments[layer],
                _weightVelocities[layer],
                _weightGradients[layer],
                learningRate,
                gradientScale,
                correction1,
                correction2);

            Update(
                _biases[layer],
                _biasMoments[layer],
                _biasVelocities[layer],
                _biasGradients[layer],
                learningRate,
                gradientScale,
                correction1,
                correction2);
        }
    }

    /// <summary>
    /// Freezes the current tensors into an inference model that can be saved as a
    /// <c>.dsbrain</c> file.
    /// </summary>
    /// <returns>A model owning copies of the training tensors.</returns>
    /// <remarks>
    /// The tensors are cloned, not handed over: <see cref="MlpModel"/> keeps a live reference to
    /// whatever it is given, and the caller checkpoints mid-training and then keeps training.
    /// </remarks>
    public MlpModel Export()
    {
        var weights = new float[_layerCount][];
        var biases = new float[_layerCount][];

        for (var layer = 0; layer < _layerCount; layer++)
        {
            weights[layer] = (float[])_weights[layer].Clone();
            biases[layer] = (float[])_biases[layer].Clone();
        }

        return MlpModel.Create(_specs, weights, biases);
    }

    /// <summary>
    /// Copies every trainable value into <paramref name="destination"/> in one flat vector, weights
    /// and biases layer by layer. This is the same order <see cref="ParameterCount"/> counts in, and
    /// the order <see cref="LoadParameters"/> reads back.
    /// </summary>
    /// <param name="destination">Buffer of at least <see cref="ParameterCount"/> floats.</param>
    /// <returns>The number of values written.</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> is too short.</exception>
    public int CopyParameters(Span<float> destination)
    {
        if (destination.Length < ParameterCount)
        {
            throw new ArgumentException(
                $"The parameter buffer must hold {ParameterCount} values, but has " +
                $"{destination.Length}.",
                nameof(destination));
        }

        var offset = 0;

        for (var layer = 0; layer < _layerCount; layer++)
        {
            _weights[layer].CopyTo(destination[offset..]);
            offset += _weights[layer].Length;
            _biases[layer].CopyTo(destination[offset..]);
            offset += _biases[layer].Length;
        }

        return offset;
    }

    /// <summary>
    /// Overwrites the trainable values from a flat vector produced by
    /// <see cref="CopyParameters"/>, and clears the Adam moments because they describe the previous
    /// weights. Used to resume a run, roll a network back, and probe gradients numerically.
    /// </summary>
    /// <param name="source">Exactly <see cref="ParameterCount"/> floats.</param>
    /// <exception cref="ArgumentException">The vector does not match the network's parameter count.</exception>
    public void LoadParameters(ReadOnlySpan<float> source)
    {
        if (source.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"The network has {ParameterCount} parameters, but the vector holds " +
                $"{source.Length}.",
                nameof(source));
        }

        var offset = 0;

        for (var layer = 0; layer < _layerCount; layer++)
        {
            source.Slice(offset, _weights[layer].Length).CopyTo(_weights[layer]);
            offset += _weights[layer].Length;
            source.Slice(offset, _biases[layer].Length).CopyTo(_biases[layer]);
            offset += _biases[layer].Length;

            Array.Clear(_weightMoments[layer]);
            Array.Clear(_weightVelocities[layer]);
            Array.Clear(_biasMoments[layer]);
            Array.Clear(_biasVelocities[layer]);
        }

        _step = 0;
    }

    /// <summary>
    /// Copies the gradient accumulated by the last <see cref="TrainSample"/> call into
    /// <paramref name="destination"/>, in the same layer order as <see cref="CopyParameters"/>.
    /// </summary>
    /// <param name="destination">Buffer of at least <see cref="ParameterCount"/> floats.</param>
    /// <returns>The number of values written.</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> is too short.</exception>
    public int CopyGradients(Span<float> destination)
    {
        if (destination.Length < ParameterCount)
        {
            throw new ArgumentException(
                $"The gradient buffer must hold {ParameterCount} values, but has " +
                $"{destination.Length}.",
                nameof(destination));
        }

        var offset = 0;

        for (var layer = 0; layer < _layerCount; layer++)
        {
            _weightGradients[layer].CopyTo(destination[offset..]);
            offset += _weightGradients[layer].Length;
            _biasGradients[layer].CopyTo(destination[offset..]);
            offset += _biasGradients[layer].Length;
        }

        return offset;
    }

    /// <summary>
    /// Overwrites the tensors with a previously exported checkpoint and clears the Adam moments, used
    /// to resume a run or roll back to the best validation loss.
    /// </summary>
    /// <exception cref="ArgumentException">The model does not match this network's shape.</exception>
    public void Import(MlpModel model)
    {
        if (model.LayerCount != _layerCount || model.InputSize != InputSize)
        {
            throw new ArgumentException(
                $"A {model.LayerCount}-layer model cannot restore a {_layerCount}-layer trainer.",
                nameof(model));
        }

        for (var layer = 0; layer < _layerCount; layer++)
        {
            model[layer].Weights.CopyTo(_weights[layer]);
            model[layer].Biases.CopyTo(_biases[layer]);
            Array.Clear(_weightMoments[layer]);
            Array.Clear(_weightVelocities[layer]);
            Array.Clear(_biasMoments[layer]);
            Array.Clear(_biasVelocities[layer]);
        }

        _step = 0;
    }

    private float MeanSquaredError(ReadOnlySpan<float> target)
    {
        var output = _activations[_layerCount];
        var sum = 0f;

        for (var i = 0; i < output.Length; i++)
        {
            var error = output[i] - target[i];
            sum += error * error;
        }

        return sum / output.Length;
    }

    private float Backward(ReadOnlySpan<float> target)
    {
        var output = _activations[_layerCount];
        var scale = 1f / output.Length;

        // Loss is the mean squared error, so the output-layer seed is dL/dy = 2 * (y - t) / n. The
        // activation derivative is folded in by the loop below, like everywhere else.
        var lastDelta = _deltas[_layerCount - 1];
        for (var i = 0; i < output.Length; i++)
        {
            lastDelta[i] = 2f * scale * (output[i] - target[i]);
        }

        var error = MeanSquaredError(target);

        for (var layer = _layerCount - 1; layer >= 0; layer--)
        {
            var activation = _specs[layer].Activation;
            var delta = _deltas[layer];
            var pre = _preActivations[layer];
            var weights = _weights[layer];
            var weightGradients = _weightGradients[layer];
            var biasGradients = _biasGradients[layer];
            var source = _activations[layer];
            var width = source.Length;
            var height = delta.Length;

            for (var outputIndex = 0; outputIndex < height; outputIndex++)
            {
                delta[outputIndex] *= ActivationMath.Derivative(pre[outputIndex], activation);
            }

            // One fused pass over the weight rows: accumulate dW and db and carry the error back onto
            // the previous layer's activations in the same walk over the tensor.
            for (var outputIndex = 0; outputIndex < height; outputIndex++)
            {
                var signal = delta[outputIndex];
                var row = outputIndex * width;
                biasGradients[outputIndex] += signal;

                for (var i = 0; i < width; i++)
                {
                    weightGradients[row + i] += signal * source[i];
                }
            }

            if (layer == 0)
            {
                continue;
            }

            // The next iteration reads this as a raw dL/dy, then multiplies by that layer's own
            // activation derivative, so nothing is applied twice here.
            var previous = _deltas[layer - 1];
            for (var i = 0; i < previous.Length; i++)
            {
                var carried = 0f;

                for (var outputIndex = 0; outputIndex < height; outputIndex++)
                {
                    carried += weights[(outputIndex * width) + i] * delta[outputIndex];
                }

                previous[i] = carried;
            }
        }

        return error;
    }

    private void Update(
        float[] values,
        float[] moments,
        float[] velocities,
        float[] gradients,
        float learningRate,
        float gradientScale,
        float correction1,
        float correction2)
    {
        var keepMoments = 1f - _beta1;
        var keepVelocities = 1f - _beta2;

        for (var i = 0; i < values.Length; i++)
        {
            var gradient = gradients[i] * gradientScale;
            moments[i] = (_beta1 * moments[i]) + (keepMoments * gradient);
            velocities[i] = (_beta2 * velocities[i]) + (keepVelocities * gradient * gradient);

            var normalized = (moments[i] / correction1) /
                ((MathF.Sqrt(velocities[i]) / correction2) + _epsilon);
            values[i] -= learningRate * normalized;
        }
    }
}
