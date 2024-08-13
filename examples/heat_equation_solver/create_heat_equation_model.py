import struct

with open("heat_equation_model.bin", "wb") as f:
    f.write(struct.pack("I", 0x67676d6c))
    
    f.write(struct.pack("i", 0))

print("Created dummy model file: heat_equation_model.bin")