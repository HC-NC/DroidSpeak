window.droidAudio = {
    lastUrl: null,

    createWavUrl: function (samples, sampleRate) {
        if (this.lastUrl) {
            URL.revokeObjectURL(this.lastUrl);
            this.lastUrl = null;
        }
        if (!samples || samples.length === 0) return "";

        const blob = this.encodeWavBlob(samples, sampleRate);
        this.lastUrl = URL.createObjectURL(blob);
        return this.lastUrl;
    },

    downloadWav: function (samples, sampleRate, fileName) {
        if (!samples || samples.length === 0) return;
        const blob = this.encodeWavBlob(samples, sampleRate);
        this.triggerDownload(blob, fileName);
    },

    downloadText: function (text, fileName) {
        const blob = new Blob([text], { type: 'application/json;charset=utf-8' });
        this.triggerDownload(blob, fileName);
    },

    triggerDownload: function (blob, fileName) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    },

    encodeWavBlob: function (samples, sampleRate) {
        const numSamples = samples.length;
        const dataSize = numSamples * 2;
        const buffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(buffer);

        const writeStr = (offset, str) => {
            for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
        };

        writeStr(0, 'RIFF');
        view.setUint32(4, 36 + dataSize, true);
        writeStr(8, 'WAVE');
        writeStr(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeStr(36, 'data');
        view.setUint32(40, dataSize, true);

        let offset = 44;
        for (let i = 0; i < numSamples; i++, offset += 2) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }

        return new Blob([buffer], { type: 'audio/wav' });
    },

    // ---- Real-time Web Audio chunked stream player ----
    streamCtx: null,
    streamAnalyser: null,
    streamNextTime: 0,
    streamSources: [],
    liveVisState: null,

    startLiveStream: function (canvasElementId) {
        this.stopLiveStream();

        const AudioCtor = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtor) return;

        // The live graph owns playback while streaming: never let the <audio> element double up.
        const player = document.getElementById('ds-audio-player');
        if (player && !player.paused) {
            player.pause();
        }

        if (!this.streamCtx || this.streamCtx.state === 'closed') {
            this.streamCtx = new AudioCtor();
            this.streamAnalyser = this.streamCtx.createAnalyser();
            this.streamAnalyser.fftSize = 2048;
            this.streamAnalyser.smoothingTimeConstant = 0.25;
            this.streamAnalyser.connect(this.streamCtx.destination);
        }

        if (this.streamCtx.state === 'suspended') {
            this.streamCtx.resume();
        }

        // Small scheduling lead so the first chunk is never late.
        this.streamNextTime = this.streamCtx.currentTime + 0.04;

        const canvas = document.getElementById(canvasElementId);
        if (canvas && this.streamAnalyser) {
            const state = {
                ctx: this.streamCtx,
                analyser: this.streamAnalyser,
                audio: { paused: false, ended: false },
                canvas: canvas,
                bins: new Uint8Array(this.streamAnalyser.frequencyBinCount),
                frame: 0,
                running: false
            };
            this.liveVisState = state;
            this.startSpectrum(state);
        }
    },

    enqueueStreamChunk: function (samples, sampleRate) {
        if (!this.streamCtx || !samples || samples.length === 0 || !sampleRate) return;

        if (this.streamCtx.state === 'suspended') {
            this.streamCtx.resume();
        }

        const buffer = this.streamCtx.createBuffer(1, samples.length, sampleRate);
        buffer.getChannelData(0).set(samples);

        const src = this.streamCtx.createBufferSource();
        src.buffer = buffer;
        src.connect(this.streamAnalyser);

        // Schedule gaplessly: each chunk starts where the previous one ends.
        const now = this.streamCtx.currentTime;
        if (this.streamNextTime < now) {
            this.streamNextTime = now + 0.01;
        }

        src.start(this.streamNextTime);
        this.streamNextTime += buffer.duration;
        this.streamSources.push(src);

        // Production outran playback, or a chunk arrived after a gap: keep the visualizer alive.
        if (this.liveVisState) {
            this.liveVisState.audio.ended = false;
            this.startSpectrum(this.liveVisState);
        }

        src.onended = () => {
            const idx = this.streamSources.indexOf(src);
            if (idx >= 0) this.streamSources.splice(idx, 1);
            if (this.streamSources.length === 0 && this.liveVisState) {
                this.liveVisState.audio.ended = true;
            }
        };
    },

    stopLiveStream: function () {
        for (const src of this.streamSources) {
            try { src.stop(); } catch (_) { }
        }
        this.streamSources = [];
        this.streamNextTime = 0;
        if (this.liveVisState) {
            this.stopSpectrum(this.liveVisState);
            this.liveVisState = null;
        }
    },

    // ---- Real-time spectrum & spectrogram visualizer (Web Audio API) ----

    spectrumBars: 48,
    visualizerMode: 'bars', // 'bars' | 'spectrogram'
    spectrogramBuffer: null,

    saveSpectrogramSnapshot: function (canvas) {
        if (!canvas) return;
        if (!this.spectrogramBuffer) {
            this.spectrogramBuffer = document.createElement('canvas');
            this.spectrogramBuffer.width = canvas.width;
            this.spectrogramBuffer.height = canvas.height;
            const bg = this.spectrogramBuffer.getContext('2d');
            if (bg) {
                bg.fillStyle = '#0b0f17';
                bg.fillRect(0, 0, canvas.width, canvas.height);
            }
        }
        const bg = this.spectrogramBuffer.getContext('2d');
        if (bg) {
            bg.drawImage(canvas, 0, 0, this.spectrogramBuffer.width, this.spectrogramBuffer.height);
        }
    },

    restoreSpectrogramSnapshot: function (canvas) {
        if (!canvas) return;
        this.clearCanvas(canvas);
        if (this.spectrogramBuffer) {
            const g = canvas.getContext('2d');
            if (g) {
                g.drawImage(this.spectrogramBuffer, 0, 0, canvas.width, canvas.height);
            }
        }
    },

    restoreSpectrogramSnapshotById: function (id) {
        this.restoreSpectrogramSnapshot(document.getElementById(id));
    },

    setVisualizerMode: function (mode, canvasElementId) {
        this.visualizerMode = mode === 'spectrogram' ? 'spectrogram' : 'bars';
        const canvas = document.getElementById(canvasElementId);
        if (canvas) {
            if (this.visualizerMode === 'bars') {
                this.drawSpectrumIdle(canvas);
            } else {
                this.restoreSpectrogramSnapshot(canvas);
            }
        }
    },

    attachVisualizer: function (audioElementId, canvasElementId) {
        const audio = document.getElementById(audioElementId);
        const canvas = document.getElementById(canvasElementId);
        if (!audio || !canvas) return;

        let state = audio.__droidSpectrum;

        if (!state) {
            const AudioCtor = window.AudioContext || window.webkitAudioContext;
            if (!AudioCtor) {
                this.drawSpectrumIdle(canvas);
                return;
            }

            try {
                const ctx = new AudioCtor();
                const source = ctx.createMediaElementSource(audio);
                const analyser = ctx.createAnalyser();
                analyser.fftSize = 2048;
                analyser.smoothingTimeConstant = 0.25;

                source.connect(analyser);
                analyser.connect(ctx.destination);

                state = {
                    ctx: ctx,
                    analyser: analyser,
                    audio: audio,
                    canvas: canvas,
                    bins: new Uint8Array(analyser.frequencyBinCount),
                    frame: 0,
                    running: false
                };
                audio.__droidSpectrum = state;

                audio.addEventListener('play', () => {
                    if (state.ctx.state === 'suspended') state.ctx.resume();
                    // if (this.visualizerMode === 'spectrogram' && audio.currentTime < 0.08) {
                    //     this.clearCanvas(state.canvas);
                    // }
                    this.startSpectrum(state);
                });
                audio.addEventListener('pause', () => {
                    this.stopSpectrum(state);
                    if (this.visualizerMode === 'bars') this.drawSpectrumIdle(state.canvas);
                });
                audio.addEventListener('ended', () => {
                    this.stopSpectrum(state);
                    if (this.visualizerMode === 'bars') this.drawSpectrumIdle(state.canvas);
                });
            } catch (err) {
                this.drawSpectrumIdle(canvas);
                return;
            }
        }

        state.canvas = canvas;

        if (state.ctx.state === 'suspended' && !audio.paused) {
            state.ctx.resume();
        }

        if (!audio.paused && !audio.ended) {
            // if (this.visualizerMode === 'spectrogram' && audio.currentTime < 0.08) {
            //     this.clearCanvas(canvas);
            // }
            this.startSpectrum(state);
        } else {
            if (this.visualizerMode === 'bars') {
                this.drawSpectrumIdle(canvas);
            } else {
                this.restoreSpectrogramSnapshot(canvas);
            }
        }
    },

    startSpectrum: function (state) {
        if (state.running) return;
        state.running = true;

        const tick = () => {
            if (!state.running) return;
            if (state.audio.paused || state.audio.ended) {
                state.running = false;
                state.frame = 0;
                if (this.visualizerMode === 'bars') {
                    this.drawSpectrumIdle(state.canvas);
                }
                return;
            }

            if (this.visualizerMode === 'spectrogram') {
                this.drawSpectrogramFrame(state);
            } else {
                this.drawSpectrumFrame(state);
            }
            state.frame = window.requestAnimationFrame(tick);
        };

        state.frame = window.requestAnimationFrame(tick);
    },

    stopSpectrum: function (state) {
        state.running = false;
        if (state.frame) {
            window.cancelAnimationFrame(state.frame);
            state.frame = 0;
        }
    },

    clearCanvas: function (canvas) {
        if (!canvas) return;
        const g = canvas.getContext('2d');
        if (!g) return;
        g.fillStyle = '#0b0f17';
        g.fillRect(0, 0, canvas.width, canvas.height);
    },

    drawSpectrumFrame: function (state) {
        const canvas = state.canvas;
        const g = canvas.getContext('2d');
        if (!g) return;

        const width = canvas.width;
        const height = canvas.height;
        const barCount = this.spectrumBars;
        const gap = 2;
        const barWidth = Math.max(2, (width - gap * (barCount - 1)) / barCount);
        const maxBarHeight = height - 4;

        state.analyser.getByteFrequencyData(state.bins);

        g.fillStyle = '#0b0f17';
        g.fillRect(0, 0, width, height);

        const usableBins = Math.max(barCount, Math.floor(state.bins.length * 0.45));

        const gradient = g.createLinearGradient(0, height, 0, 0);
        gradient.addColorStop(0, '#0ea5e9');
        gradient.addColorStop(1, '#38bdf8');

        for (let i = 0; i < barCount; i++) {
            const from = Math.floor((i * usableBins) / barCount);
            const to = Math.max(from + 1, Math.floor(((i + 1) * usableBins) / barCount));

            let peak = 0;
            for (let b = from; b < to; b++) {
                if (state.bins[b] > peak) peak = state.bins[b];
            }

            const barHeight = Math.max(2, (peak / 255) * maxBarHeight);
            const x = i * (barWidth + gap);
            const y = height - barHeight;

            g.fillStyle = gradient;
            g.fillRect(x, y, barWidth, barHeight);

            g.fillStyle = 'rgba(125, 211, 252, 0.85)';
            g.fillRect(x, y, barWidth, 2);
        }
    },

    drawSpectrogramFrame: function (state) {
        const canvas = state.canvas;
        const g = canvas.getContext('2d');
        if (!g) return;

        const width = canvas.width;
        const height = canvas.height;
        const step = 2;

        state.analyser.getByteFrequencyData(state.bins);

        // Shift existing waterfall 2px to the left
        g.drawImage(canvas, -step, 0);

        // Use lower ~55% of bins (0 Hz .. ~12 kHz) where droid whistles live
        const usableBins = Math.floor(state.bins.length * 0.55);

        for (let y = 0; y < height; y++) {
            // Bottom of canvas = low freq, top = high freq
            const normY = 1 - (y / height);
            const binIdx = Math.min(usableBins - 1, Math.floor(normY * usableBins));
            const val = state.bins[binIdx];

            g.fillStyle = this.getSpectrogramColor(val);
            g.fillRect(width - step, y, step, 1);
        }

        this.saveSpectrogramSnapshot(canvas);
    },

    getSpectrogramColor: function (v) {
        if (v < 12) return '#0b0f17';
        if (v < 70) {
            const t = (v - 12) / 58;
            return `rgb(11, ${Math.round(15 + t * 90)}, ${Math.round(23 + t * 140)})`;
        }
        if (v < 170) {
            const t = (v - 70) / 100;
            return `rgb(${Math.round(14 + t * 42)}, ${Math.round(105 + t * 84)}, ${Math.round(163 + t * 85)})`;
        }
        const t = (v - 170) / 85;
        return `rgb(${Math.round(56 + t * 199)}, ${Math.round(189 + t * 66)}, 255)`;
    },

    // ---- Microphone capture for the acoustic modem ----
    micState: null,

    startMicCapture: async function (dotNetRef) {
        if (this.micState) return this.micState.sampleRate;
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('getUserMedia is not available in this browser');
        }

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false,
                channelCount: 1
            }
        });

        const AudioCtor = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtor) {
            stream.getTracks().forEach(track => track.stop());
            throw new Error('Web Audio is not available');
        }

        const ctx = new AudioCtor();
        if (ctx.state === 'suspended') await ctx.resume();

        const source = ctx.createMediaStreamSource(stream);
        const processor = ctx.createScriptProcessor(4096, 1, 1);
        const mute = ctx.createGain();
        mute.gain.value = 0;

        const state = { ctx, stream, source, processor, mute, sampleRate: ctx.sampleRate };
        this.micState = state;

        processor.onaudioprocess = event => {
            if (this.micState !== state) return;
            const input = event.inputBuffer.getChannelData(0);
            const copy = new Float32Array(input.length);
            copy.set(input);
            dotNetRef.invokeMethodAsync('OnMicSamples', copy, state.sampleRate)
                .catch(() => { });
        };

        source.connect(processor);
        processor.connect(mute);
        mute.connect(ctx.destination);
        return state.sampleRate;
    },

    stopMicCapture: function () {
        const state = this.micState;
        if (!state) return;
        this.micState = null;

        try { state.processor.disconnect(); } catch (_) { }
        try { state.source.disconnect(); } catch (_) { }
        try { state.mute.disconnect(); } catch (_) { }
        try { state.stream.getTracks().forEach(track => track.stop()); } catch (_) { }
        try { state.ctx.close(); } catch (_) { }
    },

    drawSpectrumIdle: function (canvas) {
        if (!canvas) return;
        this.clearCanvas(canvas);
        if (this.visualizerMode === 'spectrogram') return;

        const g = canvas.getContext('2d');
        if (!g) return;

        const width = canvas.width;
        const height = canvas.height;
        const barCount = this.spectrumBars;
        const gap = 2;
        const barWidth = Math.max(2, (width - gap * (barCount - 1)) / barCount);

        g.fillStyle = 'rgba(56, 189, 248, 0.18)';
        for (let i = 0; i < barCount; i++) {
            g.fillRect(i * (barWidth + gap), height - 3, barWidth, 3);
        }
    }
};
