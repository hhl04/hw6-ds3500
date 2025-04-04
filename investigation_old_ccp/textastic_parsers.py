"""
File: textastic_parsers.py

Description: Custom parsers for the Textastic framework.
"""

import json
import re
from collections import Counter
import csv
import xml.etree.ElementTree as ET


def json_parser(filename, stop_words=None):
    """
    Parse a JSON file and extract text for analysis.
    
    Expected format: {"text": "content goes here"}
    
    Args:
        filename: Path to the JSON file
        stop_words: Set of stop words to filter out (optional)
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            
        text = raw.get('text', '')
        # Clean text (remove punctuation and convert to lowercase)
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = clean_text.split()
        
        # Store original words for comparison
        raw_words = words.copy()
        
        # Filter out stop words if provided
        if stop_words:
            words = [word for word in words if word not in stop_words]
        
        return {
            'wordcount': Counter(words),
            'numwords': len(words),
            'text': text,
            'clean_text': ' '.join(words),
            'raw_wordcount': Counter(raw_words),
            'raw_numwords': len(raw_words)
        }
    except Exception as e:
        print(f"Error parsing JSON file {filename}: {e}")
        return {
            'wordcount': Counter(),
            'numwords': 0,
            'text': "",
            'clean_text': "",
            'raw_wordcount': Counter(),
            'raw_numwords': 0
        }


def csv_parser(filename, text_column='text'):
    """
    Parse a CSV file and extract text from a specified column.
    
    Args:
        filename: Path to the CSV file
        text_column: Name of the column containing text to analyze
    """
    try:
        all_text = []
        
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if text_column in row:
                    all_text.append(row[text_column])
        
        # Combine all text
        combined_text = ' '.join(all_text)
        
        # Clean text
        clean_text = re.sub(r'[^\w\s]', '', combined_text.lower())
        words = clean_text.split()
        
        return {
            'wordcount': Counter(words),
            'numwords': len(words),
            'text': combined_text,
            'clean_text': clean_text
        }
    except Exception as e:
        print(f"Error parsing CSV file {filename}: {e}")
        return {
            'wordcount': Counter(),
            'numwords': 0,
            'text': "",
            'clean_text': ""
        }


def xml_parser(filename, text_xpath='.//text'):
    """
    Parse an XML file and extract text elements.
    
    Args:
        filename: Path to the XML file
        text_xpath: XPath to the text elements
    """
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
        
        # Extract all text elements
        text_elements = root.findall(text_xpath)
        all_text = [elem.text for elem in text_elements if elem.text]
        
        # Combine all text
        combined_text = ' '.join(all_text)
        
        # Clean text
        clean_text = re.sub(r'[^\w\s]', '', combined_text.lower())
        words = clean_text.split()
        
        return {
            'wordcount': Counter(words),
            'numwords': len(words),
            'text': combined_text,
            'clean_text': clean_text
        }
    except Exception as e:
        print(f"Error parsing XML file {filename}: {e}")
        return {
            'wordcount': Counter(),
            'numwords': 0,
            'text': "",
            'clean_text': ""
        }


def html_parser(filename):
    """
    Parse an HTML file and extract text from body.
    """
    try:
        from html.parser import HTMLParser
        
        class MyHTMLParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
                self.recording = False
            
            def handle_starttag(self, tag, attrs):
                if tag == 'body':
                    self.recording = True
            
            def handle_endtag(self, tag):
                if tag == 'body':
                    self.recording = False
            
            def handle_data(self, data):
                if self.recording and data.strip():
                    self.text.append(data.strip())
        
        # Read HTML file
        with open(filename, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Parse HTML
        parser = MyHTMLParser()
        parser.feed(html_content)
        
        # Combine all text
        combined_text = ' '.join(parser.text)
        
        # Clean text
        clean_text = re.sub(r'[^\w\s]', '', combined_text.lower())
        words = clean_text.split()
        
        return {
            'wordcount': Counter(words),
            'numwords': len(words),
            'text': combined_text,
            'clean_text': clean_text
        }
    except Exception as e:
        print(f"Error parsing HTML file {filename}: {e}")
        return {
            'wordcount': Counter(),
            'numwords': 0,
            'text': "",
            'clean_text': ""
        }