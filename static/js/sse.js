/* Incremental SSE parser. Buffers partial UTF-8 outside this class via TextDecoder. */
class SSEParser {
    constructor(onEvent) { this.buffer = ""; this.onEvent = onEvent; }
    feed(text, final = false) {
        this.buffer += text;
        let match;
        while ((match = /\r?\n\r?\n/.exec(this.buffer))) {
            const frame = this.buffer.slice(0, match.index);
            this.buffer = this.buffer.slice(match.index + match[0].length);
            const data = frame.split(/\r?\n/).filter(line => line.startsWith("data:"))
                .map(line => line.slice(5).trimStart()).join("\n");
            if (data) this.onEvent(JSON.parse(data));
        }
        if (final && this.buffer.trim()) throw new Error("流式连接意外中断");
    }
}
if (typeof module !== "undefined") module.exports = { SSEParser };
