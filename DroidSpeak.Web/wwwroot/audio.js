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
    }
};
