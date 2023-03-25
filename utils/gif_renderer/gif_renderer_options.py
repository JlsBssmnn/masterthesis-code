class Options:
    add_reverse = True
    directories = ['/epoch_7_iter_10000/fake_B', '/epoch_7_iter_10000/fake_A', '/epoch_28_iter_40000/fake_B', '/epoch_28_iter_40000/fake_A', '/epoch_49_iter_70000/fake_B', '/epoch_49_iter_70000/fake_A']
    duration = 100
    input_file = '../data/training_results/epithelial_sheets/v_1_0_10.h5'
    loop = 0
    names = ['epoch7_fakeA', 'epoch49_fakeA']
    output_dir = '../presentations/third_meeting/images/v1.0.10'
    slice_axis = 0

options = Options()
