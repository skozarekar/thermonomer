import sys

if __name__ == "__main__":
	n = len(sys.argv)
	if n == 1:
		print("first argument was: " + sys.argv[0])
	else:
		print("third argument:" + sys.argv[2])
