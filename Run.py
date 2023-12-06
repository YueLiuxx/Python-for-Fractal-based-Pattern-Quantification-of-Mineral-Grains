import MultiFractal
import os
import csv

def data_save(data, image_path):

    file_path = os.path.join('Data', f'{image_path}.csv')
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))

def figure_data_save(data, image_path, n):

    file_path = os.path.join('Data', f'{image_path}.csv')
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(n):
            data_ = data[i]
            writer.writerow(data_.keys())
            writer.writerows(zip(*data_.values()))

def multifractal(image_path,label_path):

    para = MultiFractal.fractal(image_path,label_path)
    MultiFractal.Draw(para)

    data_save(para[9], f'datasave_{image_path}_albite')
    data_save(para[10], f'imagesave_{image_path}_albite')

image_path = ''
label_path = ''
multifractal(image_path,label_path)




