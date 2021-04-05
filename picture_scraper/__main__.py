import os

path = os.getcwd()

os.chdir(path)

directory = "D:/DataWarehouse/thecarconnection"

files = ['scrape', 'tag', 'save', 'select']

if __name__ == '__main__':
    if not os.path.isdir(directory):
        os.mkdir(directory)

    [os.system('python ' + f'{file}.py ' + directory) for file in files]
