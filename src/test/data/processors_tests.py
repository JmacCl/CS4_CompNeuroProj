import unittest


class CoreDataProcessorTest(unittest.TestCase):
    def test_that_yaml_file_loads_correctly(self):
        self.assertEqual(True, False)  # add assertion here

    def test_that_error_appear_with_no_config(self):
        self.assertEqual(True, False)  # add assertion here

class TestAugmentationSupport(unittest.TestCase):

    def will_process_flipping_specification(self):
        pass

    def will_process_mix_up_specification(self):
        pass

    def will_process_rotations_specification(self):
        pass

    def will_ignore_no_specification(self):
        pass

    def will_raise_error_if_unknown_specification_appears(self):
        pass
if __name__ == '__main__':
    unittest.main()
