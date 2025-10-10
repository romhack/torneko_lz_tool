python torneko_lz_tool.py --bpp 4 d FONT.TEX 0x20C decompressed4bpp.bin
python torneko_lz_tool.py --bpp 4 c decompressed_master4bpp.bin compressed4bpp.bin
python torneko_lz_tool.py --bpp 4 d compressed4bpp.bin 0x0 redecompressed4bpp.bin
pause