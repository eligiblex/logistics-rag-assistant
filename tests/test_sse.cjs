const assert = require("node:assert/strict");
const {SSEParser} = require("../static/js/sse.js");
const events = [{type: "sources", sources: []}, {type: "token", content: "你好\n物流🚚"}, {type: "done"}];
const bytes = Buffer.from(events.map(e => "data: " + JSON.stringify(e) + "\n\n").join(""));
for (let size = 1; size <= bytes.length; size++) {
    const output = [];
    const parser = new SSEParser(event => output.push(event));
    const decoder = new TextDecoder();
    for (let start = 0; start < bytes.length; start += size)
        parser.feed(decoder.decode(bytes.subarray(start, start + size), {stream: true}));
    parser.feed(decoder.decode(), true);
    assert.deepEqual(output, events);
}
assert.throws(() => new SSEParser(() => {}).feed('data: {"type":', true));
const crlf = [];
new SSEParser(e => crlf.push(e)).feed('data: {"type":"done"}\r\n\r\n', true);
assert.deepEqual(crlf, [{type: "done"}]);
console.log("SSE UTF-8, newline, CRLF and interrupted-frame tests passed.");
