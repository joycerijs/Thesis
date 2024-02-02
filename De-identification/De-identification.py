'''Dit script kan gebruikt worden om de volgende bestanden te anonimiseren: PDF, doc, docx, odt, txt, png, tiff, jpg, jpeg.
Voornamen, achternamen, plaatsnamen, straatnamen, landen, instellingsnamen, telefoonnummers, mailadressen, BSN-nummers
en postcodes worden weggefilterd.'''

import re
import os
import glob
import yaml
import unicodedata
import pytesseract
import tika
import PyPDF2
tika.initVM()
from pdf2image import convert_from_path
from PIL import Image
from Processor import KeywordProcessor
import nl_core_news_lg as nl_nlp
from tqdm import tqdm
import zipfile
from tika import parser
from odf import text, teletype
from odf.opendocument import load


class PrivacyFilter:

    def __init__(self):
        self.keyword_processor = KeywordProcessor(case_sensitive=True)
        self.keyword_processor_names = KeywordProcessor(case_sensitive=True)
        self.url_re = None
        self.initialised = False
        self.clean_accents = True
        self.nr_keywords = 0
        self.nlp = None
        self.use_nlp = False
        self.use_wordlist = False
        self.use_re = False
        self.numbers_to_zero = False
        ##### CONSTANTS #####
        self._punctuation = ['.', ',', ' ', ':', ';', '?', '!']
        self._capture_words = ["PROPN", "NOUN"]
        self._nlp_blacklist_entities = ["WORK_OF_ART"]

    def to_string(self):
        return 'PrivacyFiter(clean_accents=' + str(self.clean_accents) + ', use_nlp=' + str(self.use_nlp) + \
               ', use_wordlist=' + str(self.use_wordlist) + ')'

    def file_to_list(self, filename, drop_first=True):
        items_count = 0
        items = []

        with open(filename, "r", encoding="utf-8") as f:
            if drop_first:
                f.readline()

            for line in f.readlines():
                items_count += 1
                line = line.rstrip()
                items.append(line)

        self.nr_keywords += items_count
        return items

    def initialize_from_file(self, filename):
        # Hiermee worden instellingen van de filter gelezen.
        with open(filename) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        clean_accents = data['clean_accents']
        nlp_filter = data['nlp_filter']
        wordlist_filter = data['wordlist_filter']
        regular_expressions = data['regular_expressions']
        datadir = data['data_directory']

        fields = {
            os.path.join(datadir, data['firstnames']): {"replacement": "<NAAM>",
                                                        "punctuation": None if nlp_filter else self._punctuation},
            os.path.join(datadir, data['lastnames']): {"replacement": "<NAAM>",
                                                       "punctuation": None if nlp_filter else self._punctuation},
            os.path.join(datadir, data['places']): {"replacement": "<PLAATS>", "punctuation": None},
            os.path.join(datadir, data['streets']): {"replacement": "<ADRES>", "punctuation": None},
            os.path.join(datadir, data['countries']): {"replacement": "<LAND>", "punctuation": None},
        }

        self.initialize(clean_accents=clean_accents,
                        nlp_filter=nlp_filter,
                        wordlist_filter=wordlist_filter,
                        regular_expressions=regular_expressions,
                        fields=fields)

    def initialize(self, clean_accents=True, nlp_filter=True, wordlist_filter=False,
                   regular_expressions=True, numbers_to_zero=False, fields=None):

        if not fields:
            fields = {
                os.path.join('datasets', 'first_names_split_goede.csv'): {"replacement": "<NAAM>",
                                                             "punctuation": None if nlp_filter else self._punctuation},
                os.path.join('datasets', 'last_names_split_goede.csv'): {"replacement": "<NAAM>",
                                                            "punctuation": None if nlp_filter else self._punctuation},
                os.path.join('datasets', 'places.csv'): {"replacement": "<PLAATS>", "punctuation": None},
                os.path.join('datasets', 'streets_Nederland.csv'): {"replacement": "<ADRES>", "punctuation": None},
                os.path.join('datasets', 'countries.csv'): {"replacement": "<LAND>", "punctuation": None},
            }

        for field in fields:
            # If there is a punctuation list, use it.
            if fields[field]["punctuation"] is not None:
                for name in self.file_to_list(field):
                    for c in self._punctuation:
                        self.keyword_processor.add_keyword(
                            "{n}{c}".format(n=name, c=c),
                            "{n}{c}".format(n=fields[field]["replacement"], c=c)
                        )
            else:
                for name in self.file_to_list(field):
                    self.keyword_processor.add_keyword(name, fields[field]["replacement"])

        if not nlp_filter:
            for name in self.file_to_list(os.path.join('datasets', 'first_names_split_goede.csv')):
                self.keyword_processor_names.add_keyword(name, "<NAAM>")

            for name in self.file_to_list(os.path.join('datasets', 'last_names_split_goede.csv')):
                self.keyword_processor_names.add_keyword(name, "<NAAM>")

        if nlp_filter:
            self.nlp = nl_nlp.load()
            self.use_nlp = True
        self.use_wordlist = wordlist_filter
        self.clean_accents = clean_accents
        self.use_re = regular_expressions
        self.numbers_to_zero = numbers_to_zero

        self.initialised = True

    @staticmethod
    def remove_email(text):
        return re.sub("(([a-zA-Z0-9_+]+(?:\.[\w-]+)*)@((?:[\w-]+\.)*\w[\w-]{0,66})\.([a-z]{2,6}(?:\.[a-z]{2})?))"
                      "(?![^<]*>)",
                      "<EMAIL>",
                      text)

    @staticmethod
    def remove_postal_codes(text):
        return re.sub(r"\b([0-9]{4}[ ]?[A-Z]{2})\b", "<POSTCODE>", text)

    @staticmethod
    def remove_phone(text):
        # return re.sub(r"\b([0-9]{10})\b", "<PHONE>", text)
        return re.sub(r"\b(\d{2}-\d{8}|\d{10})\b", "<PHONE>", text)

    @staticmethod
    def remove_bsn(text):
        return re.sub(r"\b([0-9]{9})\b", "<BSN>", text)

    @staticmethod
    def remove_accents(text):
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
        return str(text.decode("utf-8"))

    def filter_keyword_processors(self, text):
        text = self.keyword_processor.replace_keywords(text)
        text = self.keyword_processor_names.replace_keywords(text)
        return text

    def filter_regular_expressions(self, text):
        text = self.remove_email(text)
        text = self.remove_postal_codes(text)
        text = self.remove_phone(text)
        text = self.remove_bsn(text)
        return text

    @staticmethod
    def cleanup_text(result):
        # result = re.sub("<[A-Z _]+>", "<FILTERED>", result)    # deze commenten om <NAAM> te laten staan ipv <FILTERED>.
        result = re.sub(" ([ ,.:;?!])", "\\1", result)
        result = re.sub(" +", " ", result)                          # remove multiple spaces
        result = re.sub("\n +", "\n", result)                       # remove space after newline
        result = re.sub("( <FILTERED>)+", " <FILTERED>", result)    # remove multiple consecutive <FILTERED> tags
        return result.strip()

    def filter(self, text):
        if not self.initialised:
            self.initialize()
        text = str(text)
        if self.clean_accents:
            text = self.remove_accents(text)

        if self.use_re:
            text = self.filter_regular_expressions(text)

        if self.use_wordlist:
            text = self.filter_static(text)

        return self.cleanup_text(text)

    def filter_static(self, text):
        text = " " + text + " "
        text = self.filter_regular_expressions(text)
        text = self.filter_keyword_processors(text)
        return text


def filter_PDF(files_passed, input_file_path, output_file_path):
    '''input: PDF'''
    pfilter = PrivacyFilter()
    pfilter.initialize_from_file('filter.yaml')
    pytesseract.pytesseract.tesseract_cmd = r'F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Afstuderen/Thesis/Anonimiseren_data/Tesseract-OCR/tesseract.exe'
    try:
        pdfReader = PyPDF2.PdfReader(input_file_path)
        totalPages = len(pdfReader.pages)

        if totalPages > 30:
            files_passed.append(input_file_path)
            return  # Sla over voor grote PDFs
        pages = convert_from_path(input_file_path, 500, poppler_path=r'F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Afstuderen/Thesis/poppler-23.08.0/Library/bin')
        extracted_text_all = ""

        for _, img in enumerate(pages):
            extracted_text = pytesseract.image_to_string(img, lang='nld')
            extracted_text_all += extracted_text

        #extracted_text_processed = postprocess(extracted_text_all)
        anonymized_txt = pfilter.filter(extracted_text_all)

        with open(output_file_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(anonymized_txt)

    except Exception as e:
        files_passed.append(input_file_path)
        pass


def filter_image(files_passed, input_file_path, output_file_path):
    '''input: jpg, jpeg, png, tiff'''
    try:
        pfilter = PrivacyFilter()
        pfilter.initialize_from_file('filter.yaml')
        image = Image.open(input_file_path)
        pytesseract.pytesseract.tesseract_cmd = r'F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Afstuderen/Thesis/Anonimiseren_data/Tesseract-OCR/tesseract.exe'
        extracted_text = pytesseract.image_to_string(image, lang='nld')
        #extracted_text_processed = postprocess(extracted_text)
        anonymized_txt = pfilter.filter(extracted_text)

        with open(output_file_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(anonymized_txt)
    except Exception as e:
        files_passed.append(input_file_path)
        pass


def filter_word(files_passed, input_file_path, output_file_path):
    '''input: doc of docx. Let op: voor gebruik van deze functie moet Java zijn gedownload.'''
    try:
        pfilter = PrivacyFilter()
        pfilter.initialize_from_file('filter.yaml')
        parsed = parser.from_file(input_file_path)
        extracted_text = parsed['content']
        # extracted_text_processed = postprocess(extracted_text)
        anonymized_txt = pfilter.filter(extracted_text)

        with open(output_file_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(anonymized_txt)
    except Exception as e:
        files_passed.append(input_file_path)
        pass


def filter_odt(files_passed, input_file_path, output_file_path):
    try:
        pfilter = PrivacyFilter()
        pfilter.initialize_from_file('filter.yaml')
        text_doc = load(input_file_path)
        text_content = []
        for elem in text_doc.getElementsByType(text.P):
            text_content.append(teletype.extractText(elem))
        extracted_text = "\n".join(text_content)
        #extracted_text_processed = postprocess(extracted_text)
        anonymized_txt = pfilter.filter(extracted_text)
        with open(output_file_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(anonymized_txt)
    except Exception as e:
        files_passed.append(input_file_path)
        pass


def filter_txt(files_passed, input_file_path, output_file_path):
    try:
        pfilter = PrivacyFilter()
        pfilter.initialize_from_file('filter.yaml')
        extracted_text = open(input_file_path, "r")
        #extracted_text_processed = postprocess(extracted_text.read())
        anonymized_txt = pfilter.filter(extracted_text.read())
        with open(output_file_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(anonymized_txt)
    except Exception as e:
        files_passed.append(input_file_path)
        pass


def create_txt_file_path(file_path_count, file_path, output_path):
    last_subfolder = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    txt_name = f'{file_path_count}{os.path.splitext(file_name)[1]}_a.txt'
    # create subfolder in output path
    subfolder_path = os.path.join(output_path, last_subfolder)
    os.makedirs(subfolder_path, exist_ok=True)

    txt_file_path = os.path.join(subfolder_path, txt_name)
    return txt_file_path


def process_file(file_path_count, file_path, output_path, files_passed, progress_bar):
    if file_path.endswith('.pdf'):
        txt_file_path = create_txt_file_path(file_path_count, file_path, output_path)
        filter_PDF(files_passed, file_path, txt_file_path)
        progress_bar.update(1)

    if file_path.endswith(('.doc', '.docx')):
        txt_file_path = create_txt_file_path(file_path_count, file_path, output_path)
        filter_word(files_passed, file_path, txt_file_path)
        progress_bar.update(1)

    if file_path.endswith('.odt'):
        txt_file_path = create_txt_file_path(file_path_count, file_path, output_path)
        filter_odt(files_passed, file_path, txt_file_path)
        progress_bar.update(1)

    if file_path.endswith('.txt'):
        txt_file_path = create_txt_file_path(file_path_count, file_path, output_path)
        filter_txt(files_passed, file_path, txt_file_path)
        progress_bar.update(1)

    if file_path.endswith(('png', 'jpg', 'jpeg', 'tiff')):
        txt_file_path = create_txt_file_path(file_path_count, file_path, output_path)
        filter_image(files_passed, file_path, txt_file_path)
        progress_bar.update(1)
    return files_passed, progress_bar


def main():
    # Uncomment for de-identifying GHZ (ID-adults) data
    folder_path = '/GHZ'   # Pad naar de GHZ data
    output_path = '/GHZ_a'    # Let op: er moet een mapje 'GHZ_a' zijn aangemaakt
    files_passed_txt = open('files_passed.txt', 'w')

    # Uncomment for de-identifying VVT (ID-adults) data
    # folder_path = '/VVT'    # Pad naar de VVT data
    # output_path = '/VVT_a'    # Let op: er moet een mapje 'VVT_a' zijn aangemaakt
    # files_passed_txt = open('files_passed.txt', 'w')

    zip_files = glob.glob(os.path.join(folder_path, '**/*.zip'), recursive=True)
    for zip_file in zip_files:
        zip_dir = os.path.dirname(zip_file)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(zip_dir)

    files = glob.glob(os.path.join(folder_path, '**/*'))

    progress_bar = tqdm(total=len(files), unit="file")
    file_path_count = 0
    files_passed = []

    for file_path in files:
        file_path_count += 1
        files_passed, progress_bar = process_file(file_path_count, file_path, output_path, files_passed, progress_bar)
    progress_bar.close()

    for file in files_passed:
        files_passed_txt.write(file+"\n")
    files_passed_txt.close()


if __name__ == "__main__":
    print('Start')
    main()
    print('Finish')
