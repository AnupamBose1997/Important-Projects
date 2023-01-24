
import new2
import glob
import unittest

testPath = "C:/Users/ASUS/Documents/visual studio/"


class testMergedFile(unittest.TestCase):
    def test_volume(self):

        new2.getMergedFile(testPath)
        multiples = glob.glob(testPath+"op.csv")
        for file in multiples:
            self.assertTrue(file == testPath+"op.csv")


class testMergedFile2(unittest.TestCase):
    def test_volume(self):

        new2.getMergedFile(testPath)
        # multiples = glob.glob(testPath+"op.csv")
        with open(testPath+"op.csv", newline="") as f:
            count = 0
            for line in f:
                count = count+1
            self.assertTrue(count == 101)


class testMergedFile3(unittest.TestCase):
    def test_volume(self):

        new2.getMergedFile(testPath)
        # multiples = glob.glob(testPath+"op.csv")
        with open(testPath+"op.csv", newline="") as f:
            # count = 0
            for line in f:
                self.assertTrue(len(line.split(',')) == 8)


if __name__ == '__main__':
    unittest.main()
