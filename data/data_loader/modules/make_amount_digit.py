import random
import os
import pdb


class DigitGenerator(object):
    def __init__(self, full_char_path='./config/character/full_dict.txt'):
        self.digit_list = list('0123456789')
        self.full_chars = []
        self.prefix_chance = [0, 0, 0, 0, 0, 1, 1, 2, 3]
        self.suffix_chance = [0, 0, 0, 1]
        with open(full_char_path, encoding='utf-8') as f:
            skip_chars = self.digit_list + ['.']
            for line in f:
                char = line.replace('\n', '')
                if char not in skip_chars:
                    self.full_chars.append(char)

    def generate(self, max_k=6):
        text = random.sample(self.full_chars, random.choice(self.prefix_chance))
        # text += str(random.randint(0, 9))
        text += random.sample(self.digit_list, random.randint(1, max_k))
        if random.randint(1, 10) > 7:
            text += '.'
        text += random.sample(self.digit_list, random.randint(0, max_k))
        text += random.sample(self.full_chars, random.choice(self.suffix_chance))
        return ''.join(text)

    def do_make(self, num_samples=1000000,  output_path='datasets/digit/fake_digit_data.txt'):
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(num_samples):
                label = self.generate(6)
                print(label, file=f)


def make_image(label_path='datasets/digit/fake_digit_data.txt', output_dir='datasets/digit/imgs'):
    cmd = "trdg -l cn -na 2 -d 3 -c 10 -b 0 -bl 1 -rbl -f 32 -t 8 --output_dir {} -i {}".format(output_dir, label_path)
    os.system(cmd)


if __name__ == '__main__':
    # dg = DigitGenerator()
    # dg.do_make()
    make_image()


