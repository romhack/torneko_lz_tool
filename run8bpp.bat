python torneko_lz_tool.py --bpp 8 d LD.PRG 0x26C54 decompressed8bpp.bin
python torneko_lz_tool.py --bpp 8 c decompressed_master8bpp.bin compressed8bpp.bin
python torneko_lz_tool.py --bpp 8 d compressed8bpp.bin 0x0 redecompressed8bpp.bin
pause
