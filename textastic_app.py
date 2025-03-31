

from textastic import Textastic
import textastic_parsers as tp
import pprint as pp


def main():

    tt = Textastic()
    tt.load_text('file1.txt', 'A')
    tt.load_text('file2.txt', 'B')
    tt.load_text('file3.txt', 'C')
    tt.load_text('myfile.json', 'J', parser=tp.json_parser)

    pp.pprint(tt.data)


    tt.compare_num_words()

if __name__ == '__main__':
    main()

