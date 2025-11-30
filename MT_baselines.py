import math
import random
import argparse
import os
import json
from collections import Counter
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
import time
import sacrebleu
import logging
import torch.nn.functional as F
import sentencepiece as spm

# -------------------------
# grobal configs
# -------------------------
DEVICE = None

USE_AMP = False
LOGGER = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "pad": "<pad>",
    "bos": "<bos>",
    "eos": "<eos>",
    "unk": "<unk>",
}


# -------------------------
# some tool functions
# -------------------------



def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def setup_experiment_dir(args):
    if args.run_name is None:
        args.run_name = f"{args.model}_{time.strftime('%Y%m%d-%H%M%S')}"
    exp_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def setup_logging(exp_dir):
    global LOGGER
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    LOGGER.addHandler(ch)

    fh = logging.FileHandler(os.path.join(exp_dir, "train.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    LOGGER.addHandler(fh)

    LOGGER.info(f"Logging to {exp_dir}/train.log")


def plot_loss_curves(exp_dir, train_losses, val_losses, model_name):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not installed; skip plotting loss curves.")
        return

    epochs = list(range(1, len(train_losses) + 1))
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Train / Val Loss")
    plt.legend()
    path = os.path.join(exp_dir, "loss_curves.png")
    plt.savefig(path)
    plt.close()
    LOGGER.info(f"Loss curves saved to {path}")

    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_loss": train_losses,
                "val_loss": val_losses,
            },
            f,
            indent=2,
        )
    LOGGER.info(f"Loss numbers saved to {metrics_path}")


# -------------------------
# text & vocab
# -------------------------
class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing for classification (e.g. MT vocab).
    pred: [N, V], target: [N]
    """
    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        # pred: [N, V], target: [N]
        pred = pred.view(-1, pred.size(-1))
        target = target.view(-1)

        mask = target != self.ignore_index
        if mask.sum() == 0:
            return pred.new_tensor(0.0)

        pred = pred[mask]
        target = target[mask]

        log_probs = F.log_softmax(pred, dim=-1)

        true_dist = torch.full_like(log_probs,
                                    self.smoothing / (self.vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        loss = -(true_dist * log_probs).sum() / mask.sum()
        return loss


def create_criterion(args, tgt_vocab, tgt_pad_idx):
    vocab_size = len(tgt_vocab["itos"])
    if args.label_smoothing > 0.0:
        LOGGER.info(f"Using LabelSmoothingLoss, eps={args.label_smoothing}")
        return LabelSmoothingLoss(vocab_size=vocab_size,
                                  smoothing=args.label_smoothing,
                                  ignore_index=tgt_pad_idx)
    else:
        LOGGER.info("Using standard CrossEntropyLoss (no label smoothing).")
        return nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)


def simple_tokenize(text: str) -> List[str]:
    return text.strip().lower().split()

def char_tokenize(text: str) -> List[str]:
    return list(text.strip().lower())


def build_vocab(texts: List[List[str]], max_size: int = 20000, min_freq: int = 2) -> Dict[str, Dict[str, int]]:
    counter = Counter()
    for tokens in texts:
        counter.update(tokens)

    itos = [SPECIAL_TOKENS["pad"], SPECIAL_TOKENS["bos"],
            SPECIAL_TOKENS["eos"], SPECIAL_TOKENS["unk"]]

    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        if word in itos:
            continue
        itos.append(word)
        if len(itos) >= max_size:
            break

    stoi = {w: i for i, w in enumerate(itos)}
    return {"itos": itos, "stoi": stoi}


def numericalize(tokens: List[str], vocab: Dict[str, Dict[str, int]], add_bos_eos=True) -> List[int]:
    stoi = vocab["stoi"]
    ids = []
    if add_bos_eos:
        ids.append(stoi[SPECIAL_TOKENS["bos"]])
    for t in tokens:
        ids.append(stoi.get(t, stoi[SPECIAL_TOKENS["unk"]]))
    if add_bos_eos:
        ids.append(stoi[SPECIAL_TOKENS["eos"]])
    return ids


def ids_to_text(ids: List[int], vocab: Dict[str, Dict[str, int]]) -> str:
    itos = vocab["itos"]
    tokens = []
    for idx in ids:
        if idx >= len(itos):
            continue
        tok = itos[idx]
        if tok == SPECIAL_TOKENS["eos"]:
            break
        if tok in (SPECIAL_TOKENS["bos"], SPECIAL_TOKENS["pad"]):
            continue
        tokens.append(tok)

    tokenizer_type = vocab.get("tokenizer", "whitespace")

    if tokenizer_type == "sentencepiece":
        sp_model_path = vocab.get("sp_model_path", None)
        if sp_model_path is None:
            return " ".join(tokens)

        sp = spm.SentencePieceProcessor()
        sp.load(sp_model_path)
        text = sp.decode(tokens)
        return text.strip()
    elif tokenizer_type == "char":
        # char level
        return "".join(tokens).strip()
    else:
        # whitespace
        return " ".join(tokens)

def text_to_ids(text: str, vocab: Dict[str, Dict[str, int]]) -> List[int]:
    tokenizer = vocab.get("tokenizer", "whitespace")

    if tokenizer == "sentencepiece":
        sp_model_path = vocab.get("sp_model_path", None)
        if sp_model_path is None:
            raise ValueError("sp_model_path not found in vocab for sentencepiece tokenizer.")
        sp = spm.SentencePieceProcessor()
        sp.load(sp_model_path)
        tokens = sp.encode(text.strip(), out_type=str)
    elif tokenizer == "char":
        tokens = char_tokenize(text)
    else:
        tokens = simple_tokenize(text)

    return numericalize(tokens, vocab, add_bos_eos=True)


# -------------------------
# Dataset & collate
# -------------------------
class TranslationDataset(Dataset):
    def __init__(self,
                 split,
                 src_vocab,
                 tgt_vocab,
                 max_len=100,
                 src_lang="en",
                 tgt_lang="de",
                 tokenizer="whitespace",
                 extra_tgt_lang=None,
                 add_src_lang_tag=False,
                 ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.extra_tgt_lang = extra_tgt_lang
        self.add_src_lang_tag = add_src_lang_tag

        self.dataset = load_dataset(
            "IWSLT/iwslt2017",
            f"iwslt2017-{src_lang}-{tgt_lang}",
            split=split,
            trust_remote_code=True,
        )
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

        # tokenizer 信息
        self.tokenizer = tokenizer
        self.sp_src = None
        self.sp_tgt = None
        if self.tokenizer == "sentencepiece":
            src_model_path = src_vocab.get("sp_model_path", None)
            tgt_model_path = tgt_vocab.get("sp_model_path", None)
            if src_model_path is None or tgt_model_path is None:
                raise ValueError("sp_model_path not found in vocab for sentencepiece tokenizer.")
            self.sp_src = spm.SentencePieceProcessor()
            self.sp_src.load(src_model_path)
            self.sp_tgt = spm.SentencePieceProcessor()
            self.sp_tgt.load(tgt_model_path)

        self.data = []

        def _add_examples_from_hf_dataset(hf_dataset, tgt_lang_for_this_pair):
            for item in hf_dataset:
                src_text = item["translation"][self.src_lang]
                tgt_text = item["translation"][tgt_lang_for_this_pair]

                # add tag <2de> / <2fr>
                if self.add_src_lang_tag and tgt_lang_for_this_pair is not None:
                    lang_tag = f"<2{tgt_lang_for_this_pair}>"
                    src_text = f"{lang_tag} {src_text}"

                if self.tokenizer == "sentencepiece" and self.sp_src is not None:
                    src_tokens = self.sp_src.encode(src_text.strip(), out_type=str)
                    tgt_tokens = self.sp_tgt.encode(tgt_text.strip(), out_type=str)
                elif self.tokenizer == "char":
                    src_tokens = char_tokenize(src_text)
                    tgt_tokens = char_tokenize(tgt_text)
                else:
                    src_tokens = simple_tokenize(src_text)
                    tgt_tokens = simple_tokenize(tgt_text)

                if len(src_tokens) == 0 or len(tgt_tokens) == 0:
                    continue
                if len(src_tokens) > max_len or len(tgt_tokens) > max_len:
                    continue

                src_ids = numericalize(src_tokens, src_vocab, add_bos_eos=True)
                tgt_ids = numericalize(tgt_tokens, tgt_vocab, add_bos_eos=True)
                self.data.append((src_ids, tgt_ids))

        _add_examples_from_hf_dataset(self.dataset, self.tgt_lang)

        # second language
        if self.extra_tgt_lang is not None and split == "train":
            extra_dataset = load_dataset(
                "IWSLT/iwslt2017",
                f"iwslt2017-{self.src_lang}-{self.extra_tgt_lang}",
                split=split,
                trust_remote_code=True,
            )
            _add_examples_from_hf_dataset(extra_dataset, self.extra_tgt_lang)

        if split == "train":
            random.shuffle(self.data)

        # for item in self.dataset:
        #     src_text = item["translation"][self.src_lang]
        #     tgt_text = item["translation"][self.tgt_lang]

        #     if self.tokenizer == "sentencepiece" and self.sp_src is not None:
        #         src_tokens = self.sp_src.encode(src_text.strip(), out_type=str)
        #         tgt_tokens = self.sp_tgt.encode(tgt_text.strip(), out_type=str)
        #     elif self.tokenizer == "char":
        #         src_tokens = char_tokenize(src_text)
        #         tgt_tokens = char_tokenize(tgt_text)
        #     else:
        #         src_tokens = simple_tokenize(src_text)
        #         tgt_tokens = simple_tokenize(tgt_text)

        #     if len(src_tokens) == 0 or len(tgt_tokens) == 0:
        #         continue
        #     if len(src_tokens) > max_len or len(tgt_tokens) > max_len:
        #         continue

        #     src_ids = numericalize(src_tokens, src_vocab, add_bos_eos=True)
        #     tgt_ids = numericalize(tgt_tokens, tgt_vocab, add_bos_eos=True)
        #     self.data.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def pad_batch(batch, pad_idx_src, pad_idx_tgt):
    src_seqs, tgt_seqs = zip(*batch)
    src_max_len = max(len(s) for s in src_seqs)
    tgt_max_len = max(len(s) for s in tgt_seqs)

    batch_src = []
    batch_tgt = []
    for s, t in zip(src_seqs, tgt_seqs):
        s_pad = s + [pad_idx_src] * (src_max_len - len(s))
        t_pad = t + [pad_idx_tgt] * (tgt_max_len - len(t))
        batch_src.append(s_pad)
        batch_tgt.append(t_pad)

    return (
        torch.tensor(batch_src, dtype=torch.long),
        torch.tensor(batch_tgt, dtype=torch.long),
    )


# -------------------------
# RNN + Attention
# -------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths=None):
        embedded = self.dropout(self.embedding(src))  # [B, L, E]
        outputs, hidden = self.rnn(embedded)
        # outputs: [B, L, 2H]
        # hidden: [2*num_layers, B, H]

        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        hidden_cat = torch.cat([forward_hidden, backward_hidden], dim=1)
        hidden_dec_init = torch.tanh(self.fc(hidden_cat))  # [B, H]

        return outputs, hidden_dec_init.unsqueeze(0)  # [1, B, H]


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [B, H]
        # encoder_outputs: [B, L, 2H]
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [B, L, H]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [B, L, H]
        scores = self.v(energy).squeeze(2)  # [B, L]

        if mask is not None:
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask == 0, neg_inf)

        attn_weights = torch.softmax(scores, dim=1)  # [B, L]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [B, 1, 2H]
        context = context.squeeze(1)  # [B, 2H]

        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim + enc_hid_dim * 2, dec_hid_dim, batch_first=True)
        self.attention = BahdanauAttention(enc_hid_dim, dec_hid_dim)
        self.fc_out = nn.Linear(dec_hid_dim + enc_hid_dim * 2 + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask=None):
        # input: [B]
        input = input.unsqueeze(1)  # [B, 1]
        embedded = self.dropout(self.embedding(input))  # [B, 1, E]

        dec_hidden = hidden[-1]  # [B, H]
        context, attn_weights = self.attention(dec_hidden, encoder_outputs, mask)  # [B, 2H]

        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)  # [B, 1, E+2H]
        output, hidden_new = self.rnn(rnn_input, hidden)  # output: [B, 1, H]

        output = output.squeeze(1)  # [B, H]
        context = context  # [B, 2H]
        embedded = embedded.squeeze(1)  # [B, E]

        logits = self.fc_out(torch.cat((output, context, embedded), dim=1))  # [B, V]

        return logits, hidden_new, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, tgt_pad_idx, device=DEVICE):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def make_src_mask(self, src):
        mask = (src != self.src_pad_idx).to(self.device)
        return mask

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)
        src_mask = self.make_src_mask(src)

        input = tgt[:, 0]  # [B]

        for t in range(1, tgt_len):
            logits, hidden, _ = self.decoder(input, hidden, encoder_outputs, src_mask)
            outputs[:, t, :] = logits
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input = tgt[:, t] if teacher_force else top1

        return outputs


# -------------------------
# LSTM + Attention
# -------------------------
class EncoderLSTM(nn.Module):
    """
    Output:
      - encoder_outputs: [B, L, 2 * H]
      - (h0, c0): [1, B, H]
    """
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=1, dropout=0.5, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.fc = nn.Linear(hid_dim * (2 if bidirectional else 1), hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional

    def forward(self, src, src_lengths=None):
        embedded = self.dropout(self.embedding(src))  # [B, L, E]
        # outputs: [B, L, 2H]
        # hidden, cell: [num_layers * num_directions, B, H]
        outputs, (hidden, cell) = self.rnn(embedded)

        if self.bidirectional:
            forward_hidden = hidden[-2, :, :]  # [B, H]
            backward_hidden = hidden[-1, :, :]  # [B, H]
            hidden_cat = torch.cat([forward_hidden, backward_hidden], dim=1)  # [B, 2H]
        else:
            hidden_cat = hidden[-1, :, :]  # [B, H]

        hidden_dec_init = torch.tanh(self.fc(hidden_cat))  # [B, H]
        cell_dec_init = torch.zeros_like(hidden_dec_init)  # [B, H]

        # 1, B, H]
        return outputs, (hidden_dec_init.unsqueeze(0), cell_dec_init.unsqueeze(0))


class DecoderLSTM(nn.Module):
    """
    forward:
      input: [B]
      hidden: [1, B, H]
      cell:   [1, B, H]
    """
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim + enc_hid_dim * 2, dec_hid_dim, batch_first=True)
        self.attention = BahdanauAttention(enc_hid_dim, dec_hid_dim)
        self.fc_out = nn.Linear(dec_hid_dim + enc_hid_dim * 2 + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs, mask=None):
        # input: [B]
        input = input.unsqueeze(1)  # [B, 1]
        embedded = self.dropout(self.embedding(input))  # [B, 1, E]

        # hidden: [1, B, H]
        dec_hidden = hidden[-1]  # [B, H]
        context, attn_weights = self.attention(dec_hidden, encoder_outputs, mask)  # [B, 2H]

        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)  # [B, 1, E+2H]
        output, (hidden_new, cell_new) = self.rnn(rnn_input, (hidden, cell))  # output: [B, 1, H]

        output = output.squeeze(1)  # [B, H]
        embedded = embedded.squeeze(1)  # [B, E]

        logits = self.fc_out(torch.cat((output, context, embedded), dim=1))  # [B, V]

        return logits, hidden_new, cell_new, attn_weights


class Seq2SeqLSTM(nn.Module):

    def __init__(self, encoder, decoder, src_pad_idx, tgt_pad_idx, device=DEVICE):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def make_src_mask(self, src):
        mask = (src != self.src_pad_idx).to(self.device)
        return mask

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=self.device)

        encoder_outputs, (hidden, cell) = self.encoder(src)
        src_mask = self.make_src_mask(src)

        input = tgt[:, 0]  # [B]

        for t in range(1, tgt_len):
            logits, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs, src_mask)
            outputs[:, t, :] = logits

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input = tgt[:, t] if teacher_force else top1

        return outputs


# -------------------------
# Transformer
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L]


class TransformerSeq2Seq(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 src_pad_idx: int,
                 tgt_pad_idx: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def make_src_key_padding_mask(self, src):
        return (src == self.src_pad_idx)  # [B, L]

    def make_tgt_key_padding_mask(self, tgt):
        return (tgt == self.tgt_pad_idx)

    def generate_subsequent_mask(self, size: int, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    def forward(self, src, tgt, teacher_forcing_ratio: float = 0.0):
        device = src.device
        src_key_padding_mask = self.make_src_key_padding_mask(src)
        tgt_input = tgt[:, :-1]
        tgt_key_padding_mask = self.make_tgt_key_padding_mask(tgt_input)

        src_emb = self.pos_encoder(self.src_embedding(src))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt_input))

        T = tgt_input.size(1)
        tgt_mask = self.generate_subsequent_mask(T, device=device)

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # [B, T, D]

        logits = self.fc_out(out)  # [B, T, V]

        pad = torch.zeros(logits.size(0), 1, logits.size(2), device=device)
        logits = torch.cat([pad, logits], dim=1)  # [B, Lt, V]
        return logits


# -------------------------
# RNN greedy / beam
# -------------------------
@torch.no_grad()
def translate_sentence_greedy(model, src_sentence: str,
                              src_vocab, tgt_vocab,
                              max_len: int = 80) -> str:
    model.eval()
    device = DEVICE

    if isinstance(model, nn.DataParallel):
        core_model = model.module
    else:
        core_model = model

    src_ids = text_to_ids(src_sentence, src_vocab)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    encoder_outputs, hidden = core_model.encoder(src_tensor)
    src_mask = core_model.make_src_mask(src_tensor)

    bos_idx = tgt_vocab["stoi"][SPECIAL_TOKENS["bos"]]
    eos_idx = tgt_vocab["stoi"][SPECIAL_TOKENS["eos"]]

    input_token = torch.tensor([bos_idx], dtype=torch.long, device=device)
    generated_ids = [bos_idx]

    for _ in range(max_len):
        logits, hidden, _ = core_model.decoder(input_token, hidden, encoder_outputs, src_mask)
        next_token = logits.argmax(dim=-1)
        next_id = next_token.item()
        generated_ids.append(next_id)
        if next_id == eos_idx:
            break
        input_token = next_token

    return ids_to_text(generated_ids, tgt_vocab)


@torch.no_grad()
def translate_sentence_beam_search(model, src_sentence: str,
                                   src_vocab, tgt_vocab,
                                   max_len: int = 80,
                                   beam_size: int = 5,
                                   length_penalty: float = 0.7) -> str:
    model.eval()
    device = DEVICE

    if isinstance(model, nn.DataParallel):
        core_model = model.module
    else:
        core_model = model

    src_ids = text_to_ids(src_sentence, src_vocab)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    encoder_outputs, hidden = core_model.encoder(src_tensor)
    src_mask = core_model.make_src_mask(src_tensor)

    bos_idx = tgt_vocab["stoi"][SPECIAL_TOKENS["bos"]]
    eos_idx = tgt_vocab["stoi"][SPECIAL_TOKENS["eos"]]

    beams = [(0.0, [bos_idx], hidden)]

    for _ in range(max_len):
        new_beams = []
        for log_prob, seq, h in beams:
            last_id = seq[-1]
            if last_id == eos_idx:
                new_beams.append((log_prob, seq, h))
                continue

            input_token = torch.tensor([last_id], dtype=torch.long, device=device)
            logits, h_new, _ = core_model.decoder(input_token, h, encoder_outputs, src_mask)
            log_probs = torch.log_softmax(logits.squeeze(0), dim=-1)

            topk_logp, topk_idx = torch.topk(log_probs, beam_size)
            for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                new_seq = seq + [idx]
                new_log_prob = log_prob + lp
                new_beams.append((new_log_prob, new_seq, h_new))

        beams = sorted(
            new_beams,
            key=lambda x: x[0] / (len(x[1]) ** length_penalty),
            reverse=True
        )[:beam_size]

        if all(seq[-1] == eos_idx for _, seq, _ in beams):
            break

    best_log_prob, best_seq, _ = max(
        beams,
        key=lambda x: x[0] / (len(x[1]) ** length_penalty)
    )
    return ids_to_text(best_seq, tgt_vocab)


@torch.no_grad()
def translate_sentence_greedy_lstm(model, src_sentence: str,
                                   src_vocab, tgt_vocab,
                                   max_len: int = 80) -> str:

    model.eval()
    device = DEVICE

    if isinstance(model, nn.DataParallel):
        core_model = model.module
    else:
        core_model = model

    src_ids = text_to_ids(src_sentence, src_vocab)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    encoder_outputs, (hidden, cell) = core_model.encoder(src_tensor)
    src_mask = core_model.make_src_mask(src_tensor)

    bos_idx = tgt_vocab["stoi"][SPECIAL_TOKENS["bos"]]
    eos_idx = tgt_vocab["stoi"][SPECIAL_TOKENS["eos"]]

    input_token = torch.tensor([bos_idx], dtype=torch.long, device=device)
    generated_ids = [bos_idx]

    for _ in range(max_len):
        logits, hidden, cell, _ = core_model.decoder(input_token, hidden, cell, encoder_outputs, src_mask)
        next_token = logits.argmax(dim=-1)
        next_id = next_token.item()
        generated_ids.append(next_id)
        if next_id == eos_idx:
            break
        input_token = next_token

    return ids_to_text(generated_ids, tgt_vocab)


@torch.no_grad()
def translate_sentence_beam_search_lstm(model, src_sentence: str,
                                        src_vocab, tgt_vocab,
                                        max_len: int = 80,
                                        beam_size: int = 5,
                                        length_penalty: float = 0.7) -> str:

    model.eval()
    device = DEVICE

    if isinstance(model, nn.DataParallel):
        core_model = model.module
    else:
        core_model = model

    src_ids = text_to_ids(src_sentence, src_vocab)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    encoder_outputs, (hidden, cell) = core_model.encoder(src_tensor)
    src_mask = core_model.make_src_mask(src_tensor)

    bos_idx = tgt_vocab["stoi"][SPECIAL_TOKENS["bos"]]
    eos_idx = tgt_vocab["stoi"][SPECIAL_TOKENS["eos"]]

    # 每个 beam: (log_prob, seq, hidden, cell)
    beams = [(0.0, [bos_idx], hidden, cell)]

    for _ in range(max_len):
        new_beams = []
        for log_prob, seq, h, c in beams:
            last_id = seq[-1]
            if last_id == eos_idx:
                new_beams.append((log_prob, seq, h, c))
                continue

            input_token = torch.tensor([last_id], dtype=torch.long, device=device)
            logits, h_new, c_new, _ = core_model.decoder(input_token, h, c, encoder_outputs, src_mask)
            log_probs = torch.log_softmax(logits.squeeze(0), dim=-1)

            topk_logp, topk_idx = torch.topk(log_probs, beam_size)
            for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                new_seq = seq + [idx]
                new_log_prob = log_prob + lp
                new_beams.append((new_log_prob, new_seq, h_new, c_new))

        beams = sorted(
            new_beams,
            key=lambda x: x[0] / (len(x[1]) ** length_penalty),
            reverse=True
        )[:beam_size]

        if all(seq[-1] == eos_idx for _, seq, _, _ in beams):
            break

    best_log_prob, best_seq, _, _ = max(
        beams,
        key=lambda x: x[0] / (len(x[1]) ** length_penalty)
    )
    return ids_to_text(best_seq, tgt_vocab)


# -------------------------
# Transformer greedy / beam
# -------------------------
@torch.no_grad()
def translate_sentence_greedy_transformer(model, src_sentence: str,
                                          src_vocab, tgt_vocab,
                                          max_len: int = 80) -> str:
    model.eval()
    device = DEVICE

    if isinstance(model, nn.DataParallel):
        core_model = model.module
    else:
        core_model = model

    src_ids = text_to_ids(src_sentence, src_vocab)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    src_pad_idx = src_vocab["stoi"][SPECIAL_TOKENS["pad"]]
    src_key_padding_mask = (src_tensor == src_pad_idx)
    src_emb = core_model.pos_encoder(core_model.src_embedding(src_tensor))

    memory = core_model.transformer.encoder(
        src_emb,
        src_key_padding_mask=src_key_padding_mask,
    )

    bos_idx = tgt_vocab["stoi"][SPECIAL_TOKENS["bos"]]
    eos_idx = tgt_vocab["stoi"][SPECIAL_TOKENS["eos"]]

    generated = [bos_idx]

    for _ in range(max_len):
        tgt_tensor = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
        tgt_emb = core_model.pos_encoder(core_model.tgt_embedding(tgt_tensor))
        T = tgt_tensor.size(1)
        tgt_mask = core_model.generate_subsequent_mask(T, device=device)

        out = core_model.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        logits = core_model.fc_out(out[:, -1, :])
        next_id = logits.argmax(-1).item()
        generated.append(next_id)
        if next_id == eos_idx:
            break

    return ids_to_text(generated, tgt_vocab)


@torch.no_grad()
def translate_sentence_beam_search_transformer(model, src_sentence: str,
                                               src_vocab, tgt_vocab,
                                               max_len: int = 80,
                                               beam_size: int = 5,
                                               length_penalty: float = 0.7) -> str:
    model.eval()
    device = DEVICE

    if isinstance(model, nn.DataParallel):
        core_model = model.module
    else:
        core_model = model

    src_ids = text_to_ids(src_sentence, src_vocab)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    src_pad_idx = src_vocab["stoi"][SPECIAL_TOKENS["pad"]]
    src_key_padding_mask = (src_tensor == src_pad_idx)
    src_emb = core_model.pos_encoder(core_model.src_embedding(src_tensor))

    memory = core_model.transformer.encoder(
        src_emb,
        src_key_padding_mask=src_key_padding_mask,
    )

    bos_idx = tgt_vocab["stoi"][SPECIAL_TOKENS["bos"]]
    eos_idx = tgt_vocab["stoi"][SPECIAL_TOKENS["eos"]]

    beams = [(0.0, [bos_idx])]

    for _ in range(max_len):
        new_beams = []
        for log_prob, seq in beams:
            last_id = seq[-1]
            if last_id == eos_idx:
                new_beams.append((log_prob, seq))
                continue

            tgt_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            tgt_emb = core_model.pos_encoder(core_model.tgt_embedding(tgt_tensor))
            T = tgt_tensor.size(1)
            tgt_mask = core_model.generate_subsequent_mask(T, device=device)

            out = core_model.transformer.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

            logits = core_model.fc_out(out[:, -1, :])
            log_probs = torch.log_softmax(logits.squeeze(0), dim=-1)

            topk_logp, topk_idx = torch.topk(log_probs, beam_size)
            for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                new_seq = seq + [idx]
                new_log_prob = log_prob + lp
                new_beams.append((new_log_prob, new_seq))

        beams = sorted(
            new_beams,
            key=lambda x: x[0] / (len(x[1]) ** length_penalty),
            reverse=True
        )[:beam_size]

        if all(seq[-1] == eos_idx for _, seq in beams):
            break

    best_log_prob, best_seq = max(
        beams,
        key=lambda x: x[0] / (len(x[1]) ** length_penalty)
    )
    return ids_to_text(best_seq, tgt_vocab)


# -------------------------
# BLEU / chrF
# -------------------------
@torch.no_grad()
def evaluate_bleu_chrf(model, dataloader, src_vocab, tgt_vocab,
                       decode_fn,
                       max_len: int = 80):
    model.eval()

    sys_outputs = []
    refs = []

    loop = tqdm(dataloader, desc="BLEU/chrF eval", leave=False)

    for src_batch, tgt_batch in loop:
        src_batch = src_batch.to(DEVICE)
        tgt_batch = tgt_batch.to(DEVICE)

        batch_size = src_batch.size(0)

        for i in range(batch_size):
            src_ids = src_batch[i].tolist()
            src_text = ids_to_text(src_ids, src_vocab)

            pred = decode_fn(model, src_text, src_vocab, tgt_vocab, max_len=max_len)
            sys_outputs.append(pred)

            tgt_ids = tgt_batch[i].tolist()
            ref = ids_to_text(tgt_ids, tgt_vocab)
            refs.append(ref)

    bleu = sacrebleu.corpus_bleu(sys_outputs, [refs])
    chrf = sacrebleu.corpus_chrf(sys_outputs, [refs])

    LOGGER.info(f"[Eval] sacreBLEU = {bleu.score:.2f}, chrF = {chrf.score:.2f}")
    return bleu, chrf


# -------------------------
# Training / evaluation epoch
# -------------------------
def train_epoch(model, dataloader, optimizer, criterion, scaler, clip=1.0, scheduler=None, args=None):
    model.train()
    epoch_loss = 0.0
    amp_ctx = autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()

    loop = tqdm(dataloader, desc="Train", leave=False)

    for batch_idx, (src, tgt) in enumerate(loop):
        src = src.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with amp_ctx:
            output = model(src, tgt)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt_y = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt_y)

        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        
        if scheduler is not None and args is not None and args.lr_scheduler == "noam":
            scheduler.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return epoch_loss / len(dataloader)


@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0.0
    amp_ctx = autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()

    loop = tqdm(dataloader, desc="Val", leave=False)

    for src, tgt in loop:
        src = src.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)

        with amp_ctx:
            output = model(src, tgt)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt_y = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt_y)
            epoch_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    return epoch_loss / len(dataloader)


# -------------------------
# 数据构建（多语言 + 可用 ckpt vocab）
# -------------------------
def build_data(args, src_vocab=None, tgt_vocab=None):
    tokenizer = getattr(args, "tokenizer", "whitespace")
    if src_vocab is not None and "tokenizer" in src_vocab:
        tokenizer = src_vocab["tokenizer"]
        LOGGER.info(f"Tokenizer overridden by checkpoint: {tokenizer}")

    # if src_vocab is None or tgt_vocab is None:
    #     LOGGER.info(f"Building vocab from raw training data for {args.src_lang}->{args.tgt_lang} "
    #                 f"with tokenizer={tokenizer}...")

    #     raw_train = load_dataset(
    #         "IWSLT/iwslt2017",
    #         f"iwslt2017-{args.src_lang}-{args.tgt_lang}",
    #         split="train",
    #         trust_remote_code=True,
    #     )

    #     src_sentences = []
    #     tgt_sentences = []
    #     for i, item in enumerate(raw_train):
    #         src_text = item["translation"][args.src_lang].strip()
    #         tgt_text = item["translation"][args.tgt_lang].strip()
    #         src_sentences.append(src_text)
    #         tgt_sentences.append(tgt_text)
    #         if args.max_train_samples is not None and i >= args.max_train_samples:
    #             break
    if src_vocab is None or tgt_vocab is None:
        extra_tgt = getattr(args, "extra_tgt_lang", None)

        if extra_tgt is None:
            LOGGER.info(
                f"Building vocab from raw training data for {args.src_lang}->{args.tgt_lang} "
                f"with tokenizer={tokenizer}..."
            )
        else:
            LOGGER.info(
                "Building vocab from raw training data for multilingual "
                f"{args.src_lang}->{{{args.tgt_lang}, {extra_tgt}}} "
                f"with tokenizer={tokenizer}..."
            )

        src_sentences = []
        tgt_sentences = []

        # main task (src_lang -> tgt_lang)
        lang_pairs = [(args.src_lang, args.tgt_lang)]
        if extra_tgt is not None:
            lang_pairs.append((args.src_lang, extra_tgt))

        for src_lang, tgt_lang in lang_pairs:
            raw_train = load_dataset(
                "IWSLT/iwslt2017",
                f"iwslt2017-{src_lang}-{tgt_lang}",
                split="train",
                trust_remote_code=True,
            )

            for i, item in enumerate(raw_train):
                src_text = item["translation"][src_lang].strip()
                tgt_text = item["translation"][tgt_lang].strip()

                # multilingual：add a tag before source
                # e.g. <2de> / <2fr>
                if extra_tgt is not None:
                    lang_tag = f"<2{tgt_lang}>"
                    src_text = f"{lang_tag} {src_text}"

                src_sentences.append(src_text)
                tgt_sentences.append(tgt_text)

                if args.max_train_samples is not None and i >= args.max_train_samples:
                    break


        if tokenizer == "sentencepiece":
            if args.sp_model_prefix is not None:
                src_model_prefix = args.sp_model_prefix + f".{args.src_lang}"
                tgt_model_prefix = args.sp_model_prefix + f".{args.tgt_lang}"
            else:
                base = os.path.join(args.output_dir, f"spm_{args.src_lang}_{args.tgt_lang}")
                src_model_prefix = base + f".{args.src_lang}"
                tgt_model_prefix = base + f".{args.tgt_lang}"

            src_model_file = src_model_prefix + ".model"
            tgt_model_file = tgt_model_prefix + ".model"

            if not os.path.exists(src_model_file):
                LOGGER.info(f"Training SentencePiece BPE model for src: {src_model_prefix}")
                src_corpus = src_model_prefix + ".txt"
                with open(src_corpus, "w", encoding="utf-8") as f:
                    for s in src_sentences:
                        f.write(s.replace("\n", " ") + "\n")
                spm.SentencePieceTrainer.Train(
                    input=src_corpus,
                    model_prefix=src_model_prefix,
                    vocab_size=args.vocab_size,
                    model_type="bpe",
                    character_coverage=1.0,
                )

            if not os.path.exists(tgt_model_file):
                LOGGER.info(f"Training SentencePiece BPE model for tgt: {tgt_model_prefix}")
                tgt_corpus = tgt_model_prefix + ".txt"
                with open(tgt_corpus, "w", encoding="utf-8") as f:
                    for s in tgt_sentences:
                        f.write(s.replace("\n", " ") + "\n")
                spm.SentencePieceTrainer.Train(
                    input=tgt_corpus,
                    model_prefix=tgt_model_prefix,
                    vocab_size=args.vocab_size,
                    model_type="bpe",
                    character_coverage=1.0,
                )

            sp_src = spm.SentencePieceProcessor()
            sp_src.load(src_model_file)
            sp_tgt = spm.SentencePieceProcessor()
            sp_tgt.load(tgt_model_file)

            src_texts = [sp_src.encode(s, out_type=str) for s in src_sentences]
            tgt_texts = [sp_tgt.encode(s, out_type=str) for s in tgt_sentences]

            src_vocab = build_vocab(src_texts, max_size=args.vocab_size, min_freq=args.min_freq)
            tgt_vocab = build_vocab(tgt_texts, max_size=args.vocab_size, min_freq=args.min_freq)

            src_vocab["tokenizer"] = "sentencepiece"
            tgt_vocab["tokenizer"] = "sentencepiece"
            src_vocab["sp_model_path"] = src_model_file
            tgt_vocab["sp_model_path"] = tgt_model_file

            LOGGER.info(f"Vocab built with SentencePiece BPE: src={len(src_vocab['itos'])}, "
                        f"tgt={len(tgt_vocab['itos'])}")
        elif tokenizer == "char":
            src_texts = [char_tokenize(s) for s in src_sentences]
            tgt_texts = [char_tokenize(s) for s in tgt_sentences]

            src_vocab = build_vocab(src_texts, max_size=args.vocab_size, min_freq=args.min_freq)
            tgt_vocab = build_vocab(tgt_texts, max_size=args.vocab_size, min_freq=args.min_freq)

            src_vocab["tokenizer"] = "char"
            tgt_vocab["tokenizer"] = "char"

            LOGGER.info(f"Vocab built (char-level): src={len(src_vocab['itos'])}, "
                        f"tgt={len(tgt_vocab['itos'])}")
        else:
            # whitespace tokenizer
            src_texts = [simple_tokenize(s) for s in src_sentences]
            tgt_texts = [simple_tokenize(s) for s in tgt_sentences]

            src_vocab = build_vocab(src_texts, max_size=args.vocab_size, min_freq=args.min_freq)
            tgt_vocab = build_vocab(tgt_texts, max_size=args.vocab_size, min_freq=args.min_freq)

            src_vocab["tokenizer"] = "whitespace"
            tgt_vocab["tokenizer"] = "whitespace"

            LOGGER.info(f"Vocab built: src={len(src_vocab['itos'])}, tgt={len(tgt_vocab['itos'])}")
    else:
        LOGGER.info("Using vocab loaded from checkpoint.")

        # 从 ckpt vocab 中取 tokenizer 信息
        tokenizer = src_vocab.get("tokenizer", tokenizer)

    src_pad_idx = src_vocab["stoi"][SPECIAL_TOKENS["pad"]]
    tgt_pad_idx = tgt_vocab["stoi"][SPECIAL_TOKENS["pad"]]

    extra_tgt = getattr(args, "extra_tgt_lang", None)
    add_src_lang_tag = extra_tgt is not None

    train_dataset = TranslationDataset(
        "train",
        src_vocab,
        tgt_vocab,
        max_len=args.max_len,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        tokenizer=tokenizer,
        extra_tgt_lang=extra_tgt,          # add second language in training set
        add_src_lang_tag=add_src_lang_tag,
    )
    val_dataset = TranslationDataset(
        "validation",
        src_vocab,
        tgt_vocab,
        max_len=args.max_len,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        tokenizer=tokenizer,
        extra_tgt_lang=None,               # only main task: en->de
        add_src_lang_tag=add_src_lang_tag, # Still have tag
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: pad_batch(b, src_pad_idx, tgt_pad_idx),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: pad_batch(b, src_pad_idx, tgt_pad_idx),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    LOGGER.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    LOGGER.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    return src_vocab, tgt_vocab, src_pad_idx, tgt_pad_idx, train_loader, val_loader

# -------------------------
# 模型构建（超参统一）
# -------------------------
def build_model(args, src_vocab, tgt_vocab, src_pad_idx, tgt_pad_idx):
    if args.model == "rnn":
        model_name = "RNN-Attention (GRU)"
        emb_dim = args.rnn_emb
        hid_dim = args.rnn_hid
        num_layers = args.rnn_layers

        enc = Encoder(
            vocab_size=len(src_vocab["itos"]),
            emb_dim=emb_dim,
            hid_dim=hid_dim,
            num_layers=num_layers,
            dropout=args.rnn_dropout,
        )
        dec = Decoder(
            vocab_size=len(tgt_vocab["itos"]),
            emb_dim=emb_dim,
            enc_hid_dim=hid_dim,
            dec_hid_dim=hid_dim,
            dropout=args.rnn_dropout,
        )
        model = Seq2Seq(enc, dec, src_pad_idx, tgt_pad_idx, device=DEVICE).to(DEVICE)

        default_lr = 1e-3
        decode_greedy = translate_sentence_greedy
        decode_beam = translate_sentence_beam_search
        model_config = {
            "model_type": "rnn_gru",
            "src_vocab_size": len(src_vocab["itos"]),
            "tgt_vocab_size": len(tgt_vocab["itos"]),
            "emb_dim": emb_dim,
            "hid_dim": hid_dim,
            "num_layers": num_layers,
            "dropout": args.rnn_dropout,
        }

    elif args.model == "rnn_lstm":
        model_name = "RNN-Attention (LSTM)"
        emb_dim = args.rnn_emb
        hid_dim = args.rnn_hid
        num_layers = args.rnn_layers

        enc = EncoderLSTM(
            vocab_size=len(src_vocab["itos"]),
            emb_dim=emb_dim,
            hid_dim=hid_dim,
            num_layers=num_layers,
            dropout=args.rnn_dropout,
            bidirectional=True,
        )
        dec = DecoderLSTM(
            vocab_size=len(tgt_vocab["itos"]),
            emb_dim=emb_dim,
            enc_hid_dim=hid_dim,
            dec_hid_dim=hid_dim,
            dropout=args.rnn_dropout,
        )
        model = Seq2SeqLSTM(enc, dec, src_pad_idx, tgt_pad_idx, device=DEVICE).to(DEVICE)

        default_lr = 1e-3
        decode_greedy = translate_sentence_greedy_lstm
        decode_beam = translate_sentence_beam_search_lstm
        model_config = {
            "model_type": "rnn_lstm",
            "src_vocab_size": len(src_vocab["itos"]),
            "tgt_vocab_size": len(tgt_vocab["itos"]),
            "emb_dim": emb_dim,
            "hid_dim": hid_dim,
            "num_layers": num_layers,
            "dropout": args.rnn_dropout,
        }

    elif args.model == "transformer":
        model_name = "Transformer"
        d_model = args.d_model
        nhead = args.nhead
        num_encoder_layers = args.enc_layers
        num_decoder_layers = args.dec_layers
        dim_feedforward = args.ffn_dim
        dropout = args.transformer_dropout

        model = TransformerSeq2Seq(
            src_vocab_size=len(src_vocab["itos"]),
            tgt_vocab_size=len(tgt_vocab["itos"]),
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(DEVICE)

        default_lr = 3e-4
        decode_greedy = translate_sentence_greedy_transformer
        decode_beam = translate_sentence_beam_search_transformer
        model_config = {
            "model_type": "transformer",
            "src_vocab_size": len(src_vocab["itos"]),
            "tgt_vocab_size": len(tgt_vocab["itos"]),
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
        }
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    if torch.cuda.device_count() > 1:
        LOGGER.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel for {model_name}")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    lr = args.lr if args.lr is not None else default_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = create_criterion(args, tgt_vocab, tgt_pad_idx)

    return model, optimizer, criterion, decode_greedy, decode_beam, model_name, model_config
# def build_model(args, src_vocab, tgt_vocab, src_pad_idx, tgt_pad_idx):
#     if args.model == "rnn":
#         model_name = "RNN-Attention"
#         emb_dim = args.rnn_emb
#         hid_dim = args.rnn_hid
#         num_layers = args.rnn_layers

#         enc = Encoder(
#             vocab_size=len(src_vocab["itos"]),
#             emb_dim=emb_dim,
#             hid_dim=hid_dim,
#             num_layers=num_layers,
#             dropout=args.rnn_dropout,
#         )
#         dec = Decoder(
#             vocab_size=len(tgt_vocab["itos"]),
#             emb_dim=emb_dim,
#             enc_hid_dim=hid_dim,
#             dec_hid_dim=hid_dim,
#             dropout=args.rnn_dropout,
#         )
#         model = Seq2Seq(enc, dec, src_pad_idx, tgt_pad_idx, device=DEVICE).to(DEVICE)

#         default_lr = 1e-3
#         decode_greedy = translate_sentence_greedy
#         decode_beam = translate_sentence_beam_search
#         model_config = {
#             "model_type": "rnn",
#             "src_vocab_size": len(src_vocab["itos"]),
#             "tgt_vocab_size": len(tgt_vocab["itos"]),
#             "emb_dim": emb_dim,
#             "hid_dim": hid_dim,
#             "num_layers": num_layers,
#             "dropout": args.rnn_dropout,
#         }

#     elif args.model == "transformer":
#         model_name = "Transformer"
#         d_model = args.d_model
#         nhead = args.nhead
#         num_encoder_layers = args.enc_layers
#         num_decoder_layers = args.dec_layers
#         dim_feedforward = args.ffn_dim
#         dropout = args.transformer_dropout

#         model = TransformerSeq2Seq(
#             src_vocab_size=len(src_vocab["itos"]),
#             tgt_vocab_size=len(tgt_vocab["itos"]),
#             src_pad_idx=src_pad_idx,
#             tgt_pad_idx=tgt_pad_idx,
#             d_model=d_model,
#             nhead=nhead,
#             num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#         ).to(DEVICE)

#         default_lr = 3e-4
#         decode_greedy = translate_sentence_greedy_transformer
#         decode_beam = translate_sentence_beam_search_transformer
#         model_config = {
#             "model_type": "transformer",
#             "src_vocab_size": len(src_vocab["itos"]),
#             "tgt_vocab_size": len(tgt_vocab["itos"]),
#             "d_model": d_model,
#             "nhead": nhead,
#             "num_encoder_layers": num_encoder_layers,
#             "num_decoder_layers": num_decoder_layers,
#             "dim_feedforward": dim_feedforward,
#             "dropout": dropout,
#         }
#     else:
#         raise ValueError(f"Unknown model type: {args.model}")

#     if torch.cuda.device_count() > 1:
#         LOGGER.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel for {model_name}")
#         model = nn.DataParallel(model)
#     model = model.to(DEVICE)

#     lr = args.lr if args.lr is not None else default_lr
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = create_criterion(args, tgt_vocab, tgt_pad_idx)

#     return model, optimizer, criterion, decode_greedy, decode_beam, model_name, model_config


def create_scheduler(args, optimizer):
    if args.lr_scheduler == "none":
        LOGGER.info("No LR scheduler is used.")
        return None

    if args.lr_scheduler == "plateau":
        LOGGER.info(f"Using ReduceLROnPlateau scheduler: factor={args.lr_factor}, "
                    f"patience={args.lr_patience}")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            verbose=True,
        )
        return scheduler

    if args.lr_scheduler == "noam":
        warmup = args.warmup_steps

        LOGGER.info(f"Using Noam LR scheduler (warmup_steps={warmup}).")

        def lr_lambda(step):
            # step 从 0 开始，避免 0
            if step == 0:
                step = 1
            return min(step ** -0.5, step * warmup ** -1.5) * (warmup ** 0.5)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler

    raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")



# -------------------------
# ckpt 加载 helper（处理 module. 前缀）
# -------------------------
def load_model_state(model, state_dict):
    model_state = model.state_dict()
    ckpt_keys = list(state_dict.keys())
    model_keys = list(model_state.keys())

    if ckpt_keys[0].startswith("module.") and not model_keys[0].startswith("module."):
        new_state = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    elif not ckpt_keys[0].startswith("module.") and model_keys[0].startswith("module."):
        new_state = {"module." + k: v for k, v in state_dict.items()}
    else:
        new_state = state_dict

    model.load_state_dict(new_state)


# -------------------------
# 训练 & 评估（支持 resume + 画曲线）
# -------------------------
def train_and_evaluate(args,
                       model,
                       optimizer,
                       criterion,
                       best_ckpt_path,
                       decode_greedy,
                       decode_beam,
                       model_name,
                       model_config,
                       train_loader,
                       val_loader,
                       src_vocab,
                       tgt_vocab,
                       exp_dir,
                       start_epoch=0,
                       best_val_loss_init=float("inf"),
                       scaler_state=None,
                       scheduler=None):

    N_EPOCHS = args.epochs

    total_start = time.time()
    scaler = None
    if USE_AMP and DEVICE.type == "cuda":
        scaler = GradScaler(device="cuda")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
            LOGGER.info("Loaded GradScaler state from checkpoint.")


    best_val_loss = best_val_loss_init
    train_losses = []
    val_losses = []

    LOGGER.info(f"[{model_name}] Start training from epoch {start_epoch}, "
                f"total epochs={N_EPOCHS}, lr={optimizer.param_groups[0]['lr']}, amp={USE_AMP}")

    patience_counter = 0
    early_stop_patience = args.early_stop_patience
    min_delta = args.early_stop_min_delta


    for epoch in range(start_epoch, N_EPOCHS):
        train_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, scheduler=scheduler, args=args)
        train_end = time.time()

        eval_start = time.time()
        val_loss = evaluate_epoch(model, val_loader, criterion)
        eval_end = time.time()

        train_time = train_end - train_start
        eval_time = eval_end - eval_start

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 先更新 scheduler
        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_loss)
            elif args.lr_scheduler == "noam":
                pass
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        LOGGER.info(
            f"[{model_name}] Epoch {epoch+1}/{N_EPOCHS} | "
            f"train_loss={train_loss:.3f}, val_loss={val_loss:.3f} | "
            f"train_time={train_time:.2f}s, val_time={eval_time:.2f}s | "
            f"lr={current_lr:.6f}"
        )

        # early stopping
        if early_stop_patience is not None:
            if (best_val_loss - val_loss) > min_delta:
                patience_counter = 0
            else:
                patience_counter += 1
                LOGGER.info(f"[EarlyStop] No improvement. patience={patience_counter}/{early_stop_patience}")

            if patience_counter >= early_stop_patience:
                LOGGER.info("[EarlyStop] Triggered. Stopping training early.")
                break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "config": model_config,
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "epoch": epoch,
                "use_amp": USE_AMP,
                "args": vars(args),
            }
            if USE_AMP and scaler is not None:
                ckpt["scaler_state"] = scaler.state_dict()
            if scheduler is not None:
                ckpt["scheduler_state"] = scheduler.state_dict()

            torch.save(ckpt, best_ckpt_path)
            LOGGER.info(f"  -> New best model saved to {best_ckpt_path} (val_loss={best_val_loss:.3f})")

    total_end = time.time()
    LOGGER.info(f"[{model_name}] Total training time: {(total_end - total_start)/60:.2f} minutes")

    # 画曲线 + 保存 metrics
    plot_loss_curves(exp_dir, train_losses, val_losses, model_name)

    LOGGER.info(f"[{model_name}] Training finished.")
    
    # 用 best_ckpt 里的模型做最终评估
    if os.path.exists(best_ckpt_path):
        LOGGER.info(f"[{model_name}] Loading best checkpoint from {best_ckpt_path} for final BLEU/chrF evaluation.")
        best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        load_model_state(model, best_ckpt["model_state"])
    else:
        LOGGER.warning(f"[{model_name}] Best checkpoint not found at {best_ckpt_path}, using last-epoch model for eval.")

    eval_model = model
    if isinstance(model, nn.DataParallel):
        LOGGER.info(f"[{model_name}] Unwrapping DataParallel for evaluation (single-GPU decode).")
        eval_model = model.module


    LOGGER.info(f"[{model_name}] Evaluating BLEU / chrF on validation set (greedy)...")
    evaluate_bleu_chrf(
        eval_model,
        val_loader,
        src_vocab,
        tgt_vocab,
        decode_fn=decode_greedy,
        max_len=args.max_len,
    )
    LOGGER.info(f"[{model_name}] Evaluating BLEU / chrF on validation set (beam search)...")
    evaluate_bleu_chrf(
        eval_model,
        val_loader,
        src_vocab,
        tgt_vocab,
        decode_fn=lambda m, s, sv, tv, max_len: decode_beam(
            m,
            s,
            sv,
            tv,
            max_len=max_len,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
        ),
        max_len=args.max_len,
    )


    demo_sentence = "this is a small test sentence ."
    LOGGER.info(f"[{model_name}] Demo src: {demo_sentence}")

    greedy_tr = decode_greedy(
        eval_model, demo_sentence, src_vocab, tgt_vocab, max_len=args.max_len
    )
    beam_tr = decode_beam(
        eval_model, demo_sentence, src_vocab, tgt_vocab,
        max_len=args.max_len,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
    )

    LOGGER.info(f"[{model_name}] Greedy translation: {greedy_tr}")
    LOGGER.info(f"[{model_name}] Beam search translation: {beam_tr}")


# -------------------------
# 命令行参数
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="RNN-Attention & Transformer MT baseline (IWSLT17)")

    # 模型选择
    parser.add_argument("--model", type=str, default="rnn",
                        choices=["rnn", "rnn_lstm", "transformer"],
                        help="Model type: rnn (GRU) / rnn_lstm / transformer")

    # 语言对
    parser.add_argument("--src_lang", type=str, default="en",
                        help="Source language code for IWSLT17 (e.g., en)")
    parser.add_argument("--tgt_lang", type=str, default="de",
                        help="Target language code for IWSLT17 (e.g., de)")
    parser.add_argument("--extra_tgt_lang", type=str, default=None,
                help=(
                    "Optional extra target language (same src_lang) for multilingual training, "
                    "e.g. fr. If set, training data will be src_lang->{tgt_lang, extra_tgt_lang}, "
                    "but validation/BLEU only use src_lang->tgt_lang."
                ),
    )


    # 通用参数
    parser.add_argument("--amp", action="store_true",
                        help="Use GPU AMP (torch.amp.autocast + GradScaler)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size")
    parser.add_argument("--max_len", type=int, default=100,
                        help="Max sequence length for filtering dataset")
    parser.add_argument("--max_train_samples", type=int, default=100000,
                        help="Max number of training samples used to build vocab (approx)")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of DataLoader workers")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (if None, use model-specific default)")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam size for beam search demo")
    parser.add_argument("--length_penalty", type=float, default=0.7,
                        help="Length penalty for beam search")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--deterministic", action="store_true",
                        help="CuDNN deterministic mode (slower but reproducible)")

    # early stop
    parser.add_argument("--early_stop_patience", type=int, default=None,
                    help="Patience for early stopping. If None, disable early stopping.")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0,
                    help="Minimum improvement in val_loss to reset patience counter.")


    # 优化 & 正则
    parser.add_argument("--lr_scheduler", type=str, default="none",
                        choices=["none", "plateau", "noam"],
                        help="Learning rate scheduler type: none / plateau / noam (Transformer)")
    parser.add_argument("--warmup_steps", type=int, default=4000,
                        help="Warmup steps for Noam LR scheduler (Transformer)")
    parser.add_argument("--lr_factor", type=float, default=0.5,
                        help="Factor for ReduceLROnPlateau, new_lr = lr * factor")
    parser.add_argument("--lr_patience", type=int, default=2,
                        help="Patience for ReduceLROnPlateau (epochs without improvement)")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing epsilon. 0.0 disables label smoothing.")

    # BLEU 解码策略
    parser.add_argument("--eval_decode", type=str, default="greedy",
                        choices=["greedy", "beam"],
                        help="Decode method for BLEU/chrF evaluation: greedy / beam")

    # vocab
    parser.add_argument("--vocab_size", type=int, default=20000,
                        help="Max vocab size for both src and tgt")
    parser.add_argument("--min_freq", type=int, default=2,
                        help="Min frequency for vocab")

    # tokenizer 选择
    parser.add_argument("--tokenizer", type=str, default="whitespace",
                        choices=["whitespace", "sentencepiece", "char"],
                        help="Tokenizer type: whitespace, sentencepiece (BPE), or char-level")
    parser.add_argument("--sp_model_prefix", type=str, default=None,
                        help="Prefix for SentencePiece model files. "
                             "If None, will use output_dir/spm_{src}_{tgt}.{{src,tgt}}")


    # RNN 超参
    parser.add_argument("--rnn_emb", type=int, default=512,
                        help="RNN embedding dimension")
    parser.add_argument("--rnn_hid", type=int, default=1024,
                        help="RNN hidden dimension")
    parser.add_argument("--rnn_layers", type=int, default=2,
                        help="Number of GRU layers")
    parser.add_argument("--rnn_dropout", type=float, default=0.3,
                        help="Dropout for RNN encoder & decoder")

    # Transformer 超参
    parser.add_argument("--d_model", type=int, default=256,
                        help="Transformer model dimension")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--enc_layers", type=int, default=3,
                        help="Number of encoder layers")
    parser.add_argument("--dec_layers", type=int, default=3,
                        help="Number of decoder layers")
    parser.add_argument("--ffn_dim", type=int, default=1024,
                        help="Feedforward dimension")
    parser.add_argument("--transformer_dropout", type=float, default=0.1,
                        help="Dropout for Transformer")

    # output directory
    parser.add_argument("--output_dir", type=str, default="runs",
                        help="Directory to save logs and checkpoints")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name; used as subdirectory under output_dir")

    # ckpt / eval
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint path")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate (no training). Must provide --resume.")

    # GPU
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU ids for CUDA_VISIBLE_DEVICES, e.g. '0,1,2'. If None, use all visible GPUs.")

    return parser.parse_args()


# -------------------------
# main
# -------------------------
def main():
    global USE_AMP, DEVICE

    args = parse_args()

    if args.eval_only and args.resume is not None:
        resume_path = args.resume
        if os.path.isdir(resume_path):
            exp_dir_for_eval = resume_path
            candidate = os.path.join(exp_dir_for_eval, f"{args.model}_best.pt")
            if os.path.exists(candidate):
                ckpt_path = candidate
            else:
                best_ckpts = [
                    fn for fn in os.listdir(exp_dir_for_eval)
                    if fn.endswith("_best.pt")
                ]
                if not best_ckpts:
                    raise FileNotFoundError(
                        f"No *_best.pt found in directory: {exp_dir_for_eval}"
                    )
                if len(best_ckpts) > 1:
                    print(f"[WARN] Multiple *_best.pt found in {exp_dir_for_eval}, "
                          f"using {best_ckpts[0]}")
                ckpt_path = os.path.join(exp_dir_for_eval, best_ckpts[0])
        else:
            ckpt_path = resume_path
            exp_dir_for_eval = os.path.dirname(resume_path)

        args_json_path = os.path.join(exp_dir_for_eval, "args.json")
        
        if os.path.exists(args_json_path):
            with open(args_json_path, "r", encoding="utf-8") as f:
                saved_args = json.load(f)

            orig_run_name = saved_args.get("run_name", "eval")

            saved_args["resume"] = ckpt_path
            saved_args["eval_only"] = True

            if args.run_name is not None:
                saved_args["run_name"] = args.run_name
            else:
                saved_args["run_name"] = orig_run_name + "_eval"

            if args.output_dir is not None:
                saved_args["output_dir"] = args.output_dir

            if args.gpus is not None:
                saved_args["gpus"] = args.gpus

            # if hasattr(args, "beam_size"):
            #     saved_args["beam_size"] = args.beam_size
            # if hasattr(args, "length_penalty"):
            #     saved_args["length_penalty"] = args.length_penalty
            args = argparse.Namespace(**saved_args)
        else:
            print(f"[WARN] eval_only + resume={args.resume}, "
                  f"but no args.json found at {args_json_path}. "
                  f"Will use current command-line args as-is.")

    USE_AMP = args.amp

    if args.gpus is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"[INFO] Set CUDA_VISIBLE_DEVICES={args.gpus}")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if USE_AMP and not torch.cuda.is_available():
        raise RuntimeError("AMP is enabled (--amp) but no CUDA device is available.")

    exp_dir = setup_experiment_dir(args)
    setup_logging(exp_dir)

    with open(os.path.join(exp_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    LOGGER.info(f"Using device: {DEVICE}, AMP={USE_AMP}, model={args.model}")
    LOGGER.info(f"Language pair: {args.src_lang} -> {args.tgt_lang}")
    LOGGER.info(f"Experiment directory: {exp_dir}")

    set_seed(args.seed, deterministic=args.deterministic)
    LOGGER.info(f"Seed set to {args.seed}, deterministic={args.deterministic}")

    ckpt = None
    src_vocab = None
    tgt_vocab = None
    start_epoch = 0
    best_val_loss_init = float("inf")
    scaler_state = None

    if args.resume is not None:
        LOGGER.info(f"Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=DEVICE)

        src_vocab = ckpt.get("src_vocab")
        tgt_vocab = ckpt.get("tgt_vocab")

        if "best_val_loss" in ckpt:
            best_val_loss_init = ckpt["best_val_loss"]
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if ckpt.get("use_amp", False) and USE_AMP:
            scaler_state = ckpt.get("scaler_state", None)

        ckpt_model_type = ckpt.get("config", {}).get("model_type", None)
        if ckpt_model_type is not None and ckpt_model_type != args.model:
            LOGGER.warning(f"Checkpoint model_type={ckpt_model_type} but args.model={args.model}. "
                           f"Using args.model, but make sure they are intended.")

    src_vocab, tgt_vocab, src_pad_idx, tgt_pad_idx, train_loader, val_loader = build_data(
        args, src_vocab=src_vocab, tgt_vocab=tgt_vocab
    )

    (model,
     optimizer,
     criterion,
     decode_greedy,
     decode_beam,
     model_name,
     model_config) = build_model(args, src_vocab, tgt_vocab, src_pad_idx, tgt_pad_idx)

    best_ckpt_path = os.path.join(exp_dir, f"{args.model}_best.pt")

    if ckpt is not None:
        LOGGER.info("Loading model state from checkpoint...")
        load_model_state(model, ckpt["model_state"])

        if not args.eval_only and "optimizer_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                LOGGER.info("Loaded optimizer state from checkpoint.")
            except Exception as e:
                LOGGER.warning(f"Failed to load optimizer state: {e}")

    scheduler = create_scheduler(args, optimizer)
    if ckpt is not None and scheduler is not None and "scheduler_state" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
            LOGGER.info("Loaded scheduler state from checkpoint.")
        except Exception as e:
            LOGGER.warning(f"Failed to load scheduler state: {e}")

    if args.eval_only:
        if ckpt is None:
            raise ValueError("eval_only=True but no checkpoint provided via --resume.")
        LOGGER.info(f"[{model_name}] eval_only mode. Skipping training.")

        eval_model = model
        if isinstance(model, nn.DataParallel):
            LOGGER.info(f"[{model_name}] Unwrapping DataParallel for evaluation (single-GPU decode).")
            eval_model = model.module

        LOGGER.info(f"[{model_name}] Evaluating BLEU / chrF on validation set (greedy)...")
        evaluate_bleu_chrf(
            eval_model,
            val_loader,
            src_vocab,
            tgt_vocab,
            decode_fn=decode_greedy,
            max_len=args.max_len,
        )

        LOGGER.info(f"[{model_name}] Evaluating BLEU / chrF on validation set (beam search)...")
        evaluate_bleu_chrf(
            eval_model,
            val_loader,
            src_vocab,
            tgt_vocab,
            decode_fn=lambda m, s, sv, tv, max_len: decode_beam(
                m,
                s,
                sv,
                tv,
                max_len=max_len,
                beam_size=args.beam_size,
                length_penalty=args.length_penalty,
            ),
            max_len=args.max_len,
        )

        demo_sentence = "this is a small test sentence ."
        LOGGER.info(f"[{model_name}] Demo src: {demo_sentence}")

        greedy_tr = decode_greedy(
            eval_model, demo_sentence, src_vocab, tgt_vocab, max_len=args.max_len
        )
        beam_tr = decode_beam(
            eval_model, demo_sentence, src_vocab, tgt_vocab,
            max_len=args.max_len,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
        )

        LOGGER.info(f"[{model_name}] Greedy translation: {greedy_tr}")
        LOGGER.info(f"[{model_name}] Beam search translation: {beam_tr}")

        return

    train_and_evaluate(
        args,
        model,
        optimizer,
        criterion,
        best_ckpt_path,
        decode_greedy,
        decode_beam,
        model_name,
        model_config,
        train_loader,
        val_loader,
        src_vocab,
        tgt_vocab,
        exp_dir=exp_dir,
        start_epoch=start_epoch,
        best_val_loss_init=best_val_loss_init,
        scaler_state=scaler_state,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    main()
