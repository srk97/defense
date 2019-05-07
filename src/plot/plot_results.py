import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
parser = argparse.ArgumentParser(description='RESULT PLOTTING')

parser.add_argument(
    '--result_path', type=str, required=True, help='path of result file')
parser.add_argument(
    '--output_dir',
    type=str,
    default=None,
    help=
    'Output directory for the plots . Default is new directory with model name')
args = parser.parse_args()
OUTPUT_DIR = (args.result_path).split(
    '/')[-1][:-4] if args.output_dir is None else args.output_dir
if args.output_dir is None and not os.path.isdir(
    (args.result_path).split('/')[-1][:-4]):
  os.mkdir((args.result_path).split('/')[-1][:-4])


def clean_result_data(file_loc):
  with open(file_loc) as f:
    data = []
    lines = f.readlines()
    value = []
    for line in lines:
      words = line.split()
      if len(words) == 3:

        value.append(float(words[2]))
      elif len(words) == 6:
        value.append(float(words[3][:-1]))
        value.append(float(words[5]))
      elif len(words) == 4:
        value.append(float(words[3]))
      else:
        data.append(value.copy())
        value = []
    return np.array(data)


def plot_train_and_test_vs_steps(plot_data, file_loc):
  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_subplot(111)
  ax.set_xlabel('STEPS')
  ax.set_ylabel('ACCURACY')
  plt.plot(plot_data[:, 0], plot_data[:, 1] * 100, 'r', label='train')
  plt.plot(plot_data[:, 0], plot_data[:, 3], 'b', label='test')
  ax.legend(loc='best')
  plt.savefig(OUTPUT_DIR + '/' + 'test_and_train_vs_steps' +
              (file_loc).split('/')[-1][:-4] + '.png')
  plt.clf()


def plot_loss_vs_steps(plot_data, file_loc):
  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_subplot(111)
  ax.set_xlabel('STEPS')
  ax.set_ylabel('LOSS')
  plt.plot(plot_data[:, 0], plot_data[:, 2], 'b', label='test')
  plt.savefig(OUTPUT_DIR + '/' + 'loss_vs_steps' +
              (file_loc).split('/')[-1][:-4] + '.png')
  plt.clf()


if __name__ == "__main__":
  plot_data = clean_result_data(args.result_path)
  plot_train_and_test_vs_steps(plot_data, args.result_path)
  plot_loss_vs_steps(plot_data, args.result_path)