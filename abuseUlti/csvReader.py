import csv


class CSV_Reader(object):
    def __init__(self, file_path, delimiter=',', quotechar='"', skip_row=1, text_field=2, label_field=None, text_callback=None, label_callback=None, return_text_only=False):
        self.file_path = file_path
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.skip_row = skip_row
        self.text_field = text_field
        self.label_field = label_field
        self.text_callback = text_callback
        self.label_callback = label_callback
        self.return_text_only = return_text_only

        self.num_lines = self._get_len()
        self.csv_file = None
        self.csv_Reader = None
        self.reset()
        

    def _get_len(self):
        numLines = 0
        with open(self.file_path, 'r') as csvfile:
            csvReader = csv.reader(csvfile, delimiter=self.delimiter, quotechar=self.quotechar)
            for item in csvReader:
                numLines += 1
            numLines = numLines - self.skip_row
        return numLines

    def __iter__(self):
        self.doc_id = 0
        try:
            self.csv_file.close()
        except:
            pass
        self.csv_file = open(self.file_path, 'r')
        self.csv_Reader = csv.reader(self.csv_file, delimiter=self.delimiter, quotechar=self.quotechar)
        for i in range(self.skip_row):
            next(self.csv_Reader)
        return self

    def __next__(self):
        if self.doc_id < self.num_lines:
            row = next(self.csv_Reader)
            text = row[self.text_field]
            label = None
            if self.label_field != None:
                label = row[self.label_field]

            if self.text_callback:
                text = self.text_callback(text)
            if self.label_callback:
                label = self.label_callback(label)
            self.doc_id+=1
            if self.return_text_only:
                return text
            else:
                return text, label
        
        else:
            raise StopIteration
        
    def __len__(self):
        return self.num_lines

    def reset(self):
        self.doc_id = 0
        try:
            self.csv_file.close()
        except:
            pass
        self.csv_file = open(self.file_path, 'r')
        self.csv_Reader = csv.reader(self.csv_file, delimiter=self.delimiter, quotechar=self.quotechar)
        for i in range(self.skip_row):
            next(self.csv_Reader)


               



