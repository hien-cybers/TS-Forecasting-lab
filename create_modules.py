from pathlib import Path

src_dir = Path("src")
src_dir.mkdir(exist_ok=True)

# 1. Tạo file __init__.py
(src_dir / "__init__.py").write_text("# src package\n", encoding="utf-8")

print("[THÀNH CÔNG] Đã chuẩn bị xong thư mục src và file __init__.py")