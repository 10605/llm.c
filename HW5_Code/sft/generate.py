"""Generate one sequence from one prompt, then exit (no interactive loop).

    python generate.py --model model.bin --prompt "The experiment showed"
    python generate.py --model sft.bin --mode chat --prompt "Explain gravity."

"chat" selects the SFT prompt format; it still generates just one response.
"""

import argparse
from contextlib import nullcontext


def build_parser():
    parser = argparse.ArgumentParser(description="Generate one sequence from an llm.c checkpoint")
    parser.add_argument("--model", required=True, help="llm.c model .bin")
    parser.add_argument("-p", "--prompt", required=True, help="one prompt; generate once and exit")
    parser.add_argument("--mode", default="auto", choices=["auto", "chat", "completion"],
                        help="prompt format; auto detects nonzero chat-token embeddings")
    parser.add_argument("-t", "--temperature", type=float, default=0.6, help="0 = greedy")
    parser.add_argument("-k", "--top-k", type=int, default=50)
    parser.add_argument("-g", "--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42, help="sampling seed (-1 = random)")
    parser.add_argument("--device", default="", help="cuda|mps|cpu (empty = autodetect)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0 or args.temperature < 0 or args.top_k < 0:
        parser.error("max-new-tokens must be positive; temperature and top-k must be nonnegative")

    import torch
    from chat_tokenizer import ChatTokenizer
    from gpt2 import load_llmc

    device = args.device or ("cuda" if torch.cuda.is_available() else
                            "mps" if torch.backends.mps.is_available() else "cpu")
    device_type = torch.device(device).type
    if args.seed >= 0:
        torch.manual_seed(args.seed)
    tokenizer = ChatTokenizer()
    model = load_llmc(args.model, device).eval()
    cfg = model.config
    if cfg.vocab_size != tokenizer.base_vocab_size:
        parser.error("this generator requires the GPT-2 tokenizer vocabulary")
    mode = args.mode
    if mode == "auto":
        chat_rows = model.wte.weight[cfg.vocab_size:tokenizer.n_vocab]
        mode = "chat" if chat_rows.abs().sum().item() > 0 else "completion"
    if mode == "chat":
        if tokenizer.n_vocab > cfg.padded_vocab_size:
            parser.error("checkpoint has no room for the chat tokens")
        model.n_active = tokenizer.n_vocab
        ids = tokenizer.render_for_completion([{"role": "user", "content": args.prompt}])
        stop = tokenizer.special("<|assistant_end|>")
    else:
        ids = [tokenizer.bos] + tokenizer.encode(args.prompt)
        stop = tokenizer.bos
    if len(ids) + args.max_new_tokens > cfg.max_seq_len:
        parser.error(f"prompt has {len(ids)} tokens; at most {cfg.max_seq_len - len(ids)} new tokens "
                     f"fit in the checkpoint's {cfg.max_seq_len}-token context")

    autocast = (torch.autocast(device_type, dtype=torch.bfloat16)
                if device_type in ("cuda", "mps") else nullcontext())
    if mode == "completion":
        print(args.prompt, end="", flush=True)
    with autocast:
        tokens = model.generate_stream(ids, args.max_new_tokens, stop_token=stop,
                                       temperature=args.temperature, top_k=args.top_k)
        for piece in tokenizer.stream_text(tokens, stop):
            print(piece, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
