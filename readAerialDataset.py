import os
import random
import glob


def read_dataset(data_dir):
    directories = ['training', 'validation']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(data_dir, "sat", directory, '*.' + 'tiff')
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
                annotation_file = os.path.join(data_dir, "map", directory, filename + '.tif')
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)
        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))

        return image_list['training'], image_list['validation']




import BatchDatsetReader as dataset


# testing area ....
train_records, valid_records = read_dataset("/Users/taafoso2/Documents/swisscom/hackathon/FCN.tensorflow/data")

image_options = {'resize': True, 'resize_size': 1500}
train_dataset_reader = dataset.BatchDatset(train_records, image_options)
train_images, train_annotations = train_dataset_reader.next_batch(1)

print('end')