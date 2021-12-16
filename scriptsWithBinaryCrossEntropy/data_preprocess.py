import csv

with open("../jigsaw-toxic-severity-rating/validation_data.csv") as f:
    reader = csv.reader(f)
    less_toxics, more_toxics, workers = [], [], []
    next(reader)
    for row in reader:
        less_toxics.append(row[1])
        more_toxics.append(row[2])
        workers.append(row[0])
with open("../jigsaw-toxic-severity-rating/myData.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['worker', 'text1', 'text2', 'label'])
    for i in range(len(less_toxics)):
        if i % 2 == 0:
            writer.writerow([workers[i], more_toxics[i], less_toxics[i], 1])
        else:
            writer.writerow([workers[i], less_toxics[i], more_toxics[i], 0])
