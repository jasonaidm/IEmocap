import string
path = "/Users/xinyu/Desktop/text_output_new.txt"
f = open(path, 'r')
w_str = ""
punctuation = string.punctuation.replace("'", '')  # !"#$%&()*+,-./:;<=>?@[\]^_`{|}~
identify = ' ' * 31
table = str.maketrans(punctuation, identify)
for line in f:
    text = line.translate(table).replace('   ', ' ').replace('  ',' ')
    w_str += text
w = open(path, 'w')
w.write(w_str)
f.close()
w.close()
