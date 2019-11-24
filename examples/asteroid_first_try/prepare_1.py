"""
takes 2 files on input - settings and template

settings:
 column 1 - name of the output file
 column 2-END - info to be pasted to template
"""
import sys
import os


settings_file = sys.argv[1]
template_file = sys.argv[2]

with open(template_file) as template_file_:
    template = template_file_.read()

with open(settings_file) as in_file:
    for line in in_file.readlines():
        file_name = line.split()[0]
        fields = line.split()[1:]
        if os.path.isfile(file_name):
            raise ValueError('File exists: {:}'.format(file_name))
        with open(file_name, 'w') as out_file:
            out_file.write(template.format(*fields))
