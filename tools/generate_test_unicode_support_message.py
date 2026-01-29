import re

class verifier:
	def verify(data):
		found = re.findall(r'U\+([0-9a-f]{6}): \"(.*)\"', data)
		for (order, character) in found:
			if len(character) == 1 and int.from_bytes(bytes.fromhex(order)) == ord(character):
				print(f"U+{order}: \"{character}\" ✅")
			else:
				print(f"U+{order}: \"{character}\" ❌")

def print_test(order):
	print(f"U+{format(order,'06x')}: \"{chr(order)}\"")

# White Spaces
print('-'*10)
print('White Spaces')
print_test(0x0020)
print_test(0x00a0)
print_test(0x180e)
for i in range(0x2000, 0x200A+1):
	print_test(i)
print_test(0x202f)
print_test(0x205f)
print_test(0x3000)

# Zero-width characters
print('-'*10)
print('Zero-Width characters')

print_test(0x200c)
print_test(0x200b)
print_test(0x200d)
for i in range(0x2060, 0x2064+1):
	print_test(i)
for i in range(0x206a, 0x206f+1):
	print_test(i)
print_test(0xfeff)
print_test(0x202c)
print_test(0x202d)
print_test(0x200e)

# Unicodeblock Variantselector
print('-'*10)
print('Variant selectors')
# 	First 16:
for i in range(0xfe00, 0xfe0f+1):
	print_test(i)
# 	17+
for i in range(0xe0100, 0xe01ef+1):
	print_test(i)

# Unicodeblock Tags
print('-'*10)
print('Tags')
for i in range(0xe0000, 0xe007f+1):
	print_test(i)

