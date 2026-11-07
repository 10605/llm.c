"""
GPT-2 tokenizer extended with chat special tokens, in the style of nanochat.

The GPT-2 vocab has 50257 tokens but llm.c pads the embedding table to 50304 rows,
so we place the chat tokens in the unused rows 50257, 50258, ... without resizing
the model. <|endoftext|> (50256) already starts every document in pretraining, so
we reuse it as the beginning-of-conversation token.
"""

import tiktoken

SPECIAL_TOKENS = [
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
]


class ChatTokenizer:
    def __init__(self, base_vocab_size=50257):
        gpt2 = tiktoken.get_encoding("gpt2")
        special = {"<|endoftext|>": gpt2.eot_token}
        for i, tok in enumerate(SPECIAL_TOKENS):
            special[tok] = base_vocab_size + i
        self.enc = tiktoken.Encoding(
            name="gpt2_chat",
            pat_str=gpt2._pat_str,
            mergeable_ranks=gpt2._mergeable_ranks,
            special_tokens=special,
        )
        self.bos = gpt2.eot_token
        self.base_vocab_size = base_vocab_size
        self.n_vocab = base_vocab_size + len(SPECIAL_TOKENS)

    def special(self, name):
        return self.enc._special_tokens[name]

    def encode(self, text):
        return self.enc.encode_ordinary(text)

    def decode(self, ids):
        return self.enc.decode(ids)

    def _normalize(self, messages):
        """Fold a leading system message into the first user message, check alternation."""
        messages = list(messages)
        if messages and messages[0]["role"] == "system":
            assert len(messages) > 1 and messages[1]["role"] == "user"
            merged = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = [{"role": "user", "content": merged}] + messages[2:]
        for i, m in enumerate(messages):
            expected = "user" if i % 2 == 0 else "assistant"
            assert m["role"] == expected, f"message {i} has role {m['role']}, expected {expected}"
        return messages

    def render_conversation(self, messages, max_tokens=1025):
        """
        Returns (ids, mask). mask[i] = 1 if token i is one the model should learn to
        produce (assistant content and <|assistant_end|>), else 0.
        """
        messages = self._normalize(messages)
        user_start, user_end = self.special("<|user_start|>"), self.special("<|user_end|>")
        asst_start, asst_end = self.special("<|assistant_start|>"), self.special("<|assistant_end|>")
        ids, mask = [self.bos], [0]

        def add(tokens, m):
            ids.extend(tokens)
            mask.extend([m] * len(tokens))

        for m in messages:
            content = self.encode(m["content"])
            if m["role"] == "user":
                add([user_start], 0)
                add(content, 0)
                add([user_end], 0)
            else:
                add([asst_start], 0)
                add(content, 1)
                add([asst_end], 1)
        return ids[:max_tokens], mask[:max_tokens]

    def render_for_completion(self, messages):
        """Render all but the final assistant turn, then prime the assistant to respond."""
        messages = list(messages)
        if messages and messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        ids, _ = self.render_conversation(messages, max_tokens=10**9)
        return ids + [self.special("<|assistant_start|>")]

    def render_to_fit(self, messages, max_new_tokens, max_seq_len=1024):
        """render_for_completion, dropping the oldest exchanges until the reply has room."""
        messages = list(messages)
        while True:
            ids = self.render_for_completion(messages)
            if len(ids) + max_new_tokens <= max_seq_len or len(messages) <= 1:
                return ids[-(max_seq_len - max_new_tokens):]
            messages = messages[2:]  # drop the oldest user/assistant exchange

    def stream_text(self, tokens, stop_token):
        """Turn a stream of generated token ids into a stream of text pieces, ending at stop_token."""
        out, sent = [], ""
        for tok in tokens:
            if tok == stop_token:
                break
            out.append(tok)
            text = self.decode([t for t in out if t < self.base_vocab_size])
            if text.endswith("�"):  # wait for the rest of a multi-byte character
                continue
            yield text[len(sent):]
            sent = text
