"""markitdown + 切片 调试脚本：给一个文件路径，看它转出的 markdown 和切片结果。

用法：
    python -m Agent.test.markitdown_debug <file_path>

例如：
    python -m Agent.test.markitdown_debug some.pdf
    python -m Agent.test.markitdown_debug photo.png
"""

import os
import sys


def main():


    path = "任政-表决票.pdf"
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return

    # 1) markitdown 转成 markdown 文本
    from markitdown import MarkItDown
    result = MarkItDown().convert(path)
    text = result.text_content

    print(f"文件: {path}")
    print(f"转换后字符数: {len(text)}")
    print("=== markdown 全文 ===")
    print(text)
    print("=== markdown 结束 ===\n")

    # 2) 喂给 pipeline 切片，看段落和 heading_path
    from Agent.Memory.rag.pipeline import (_split_paragraphs_with_headings,
                                           _chunk_paragraphs)
    paras = _split_paragraphs_with_headings(text)
    print(f"段落数: {len(paras)}")
    for i, p in enumerate(paras):
        head = p["heading_path"] or "(无标题)"
        snippet = p["content"].replace("\n", " ")[:80]
        print(f"  [{i}] {head}: {snippet}")

    chunks = _chunk_paragraphs(paras, chunk_tokens=800, overlap_tokens=100)
    print(f"\n最终 chunk 数: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  [chunk {i}] ({len(c['content'])} 字) {c['content'][:80].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
