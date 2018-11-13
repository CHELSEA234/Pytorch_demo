import argparse

parser = argparse.ArgumentParser()
# input type should be int, help is sentence you are going to output, 'integer' means args.integer will be used
# if you input -h, the help message will be output
parser.add_argument('--integer', type=int, help='display an integer')	# '--integer' means you should include pre_fix
parser.add_argument("--square", help="display a square of a given number", type=int)
parser.add_argument("--cubic", help="display a cubic of a given number", type=int)
args = parser.parse_args()

print (args.integer)		# if you don't input, this will output None
print (args.square**2)
print (args.cubic**3)
