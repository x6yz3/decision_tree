
import numpy as np
import matplotlib.pyplot as plt

clean_data = np.loadtxt('co395-cbc-dt/wifi_db/clean_dataset.txt')
noisy_data = np.loadtxt('co395-cbc-dt/wifi_db/noisy_dataset.txt')


# statistic number of components of each label in data-set
def sum_label(dataset):
    sum_1 = [0, 0, 0, 0]
    for i in range(dataset.shape[0]):
        if dataset[i][7] == 1:
            sum_1[0] += 1
        if dataset[i][7] == 2:
            sum_1[1] += 1
        if dataset[i][7] == 3:
            sum_1[2] += 1
        if dataset[i][7] == 4:
            sum_1[3] += 1
    return sum_1


# split input data into training and validation set
def split_data(data):
    size, _ = data.shape
    split_point = int((1. / 10) * (size))

    np.random.shuffle(data)

    test_data = data[0:split_point]
    train_data = data[split_point:size]
    return train_data, test_data


def calentropy(dataset, sum_array):
    class_labels = np.unique(dataset[:, 7])
    # print dataset.shape, class_labels.shape
    H = 0
    for label in class_labels:
        # print sum_array[int(label)-1]
        pk = sum_array[int(label) - 1] / float(dataset.shape[0])
        if pk != 0:
            h = pk * np.log2(pk)
            H -= h
    return H


def find_split(training_dataset):
    sum_array = sum_label(training_dataset)
    # print sum_array
    dataset_entropy = calentropy(training_dataset, sum_array)
    max_Gain = {'Wifi': 0, 'Cut_point': 0.0, 'Gain': 0.0}

    for wifi in range(7):
        wifi_length_unique = np.unique(training_dataset[:, wifi])
        # wifi_midpoint=[]
        # for i in range(len(wifi_length_unique)-1):
        #     midpoint = (wifi_length_unique[i]+ wifi_length_unique[i+1])/2
        #     wifi_midpoint.append(midpoint)
        # print wifi_length_unique
        for cut_point in wifi_length_unique:
            left_index = np.where(training_dataset[:, wifi] <= cut_point)
            right_index = np.where(training_dataset[:, wifi] > cut_point)
            left_dataset = training_dataset[left_index]
            left_sum_array = sum_label(left_dataset)
            right_dataset = training_dataset[right_index]
            right_sum_array = sum_label(right_dataset)
            remainder = left_dataset.shape[0] / float(training_dataset.shape[0]) * calentropy(left_dataset,
                                                                                              left_sum_array) \
                        + right_dataset.shape[0] / float(training_dataset.shape[0]) * calentropy(right_dataset,
                                                                                                 right_sum_array)
            Gain = dataset_entropy - remainder
            # print Gain
            if Gain > max_Gain['Gain']:
                # wifi: 0-6
                max_Gain['Wifi'] = wifi + 1
                max_Gain['Cut_point'] = cut_point
                max_Gain['Gain'] = Gain
    return max_Gain


###########################################TRAINING###########################################


# recursive function
def decision_tree_learning(training_dataset, depth):
    if len(np.unique(training_dataset[:, 7])) == 1:
        # leaf
        label = np.unique(training_dataset[:, 7])
        # number of points in this node
        num, _ = training_dataset.shape

        #
        # if dt is a leaf, then the value of 'isLeaf' is its final label
        #
        dt = {'Attribute': 0, 'Value': 0, 'Left': None, 'Right': None, 'isLeaf': int(label), 'Num': num}
        return dt, depth
    else:
        split = find_split(training_dataset)
        cut_point = split['Cut_point']
        # 1-7
        wifi = split['Wifi']
        left_index = np.where(training_dataset[:, wifi - 1] <= cut_point)
        right_index = np.where(training_dataset[:, wifi - 1] > cut_point)
        left_dataset = training_dataset[left_index]
        right_dataset = training_dataset[right_index]
        left_branch, left_depth = decision_tree_learning(left_dataset, depth + 1)
        right_branch, right_depth = decision_tree_learning(right_dataset, depth + 1)

        num = left_branch['Num'] + right_branch['Num']

        #
        # if dt is not leaf, then the value of 'isLeaf' is 0, 'Attribute' its cut wifi, 'Value' its cut position
        #
        dt = {'Attribute': wifi, 'Value': cut_point, 'Left': left_branch, 'Right': right_branch, 'isLeaf': 0,
              'Num': num}
        return dt, max(left_depth, right_depth)
        # print find_split(clean_data)


# use validate point to evaluate
def statistic_data(test_data, dt):
    confusion_matrix = np.zeros([4, 4])
    for point in test_data:
        predict, actual = validate_label(point, dt)
        confusion_matrix[int(predict) - 1][int(actual) - 1] += 1

    return confusion_matrix


# given a point, return predict label (acquired by tree) and actual label of it
def validate_label(point, dt):
    predict_label = 0
    current_node = dt
    # if not leaf
    while current_node.get('isLeaf') == 0:
        current_cutpoint = current_node.get('Value')
        current_wifi = current_node.get('Attribute')
        current_value = point[current_wifi - 1]
        if current_value <= current_cutpoint:
            current_node = current_node.get('Left')
        else:
            current_node = current_node.get('Right')

        predict_label = current_node.get('isLeaf')

    return predict_label, point[7]


###########################################PRUNING###########################################


# if two children of this nodes are leaf, check if delete this two children
def check_if_delete_child(valid_data, dt, dir):
    # for each point in data set, only those that comes to this direction will affect outcome
    useful_valid_data = []
    for point in valid_data:
        not_useful = 0
        # [0] Attribute
        # [1] Value
        # [2] which branch
        for constraint in dir:
            if ((point[constraint[0] - 1] > constraint[1]) and constraint[2] == 'left') \
            or ((point[constraint[0] - 1] <= constraint[1]) and constraint[2] == 'right'):
                not_useful = 1

        # if not constraint,
        if not_useful == 0:
            useful_valid_data.append(point)

    # number of useful valid data in this node
    # print len(valid_data), len(useful_valid_data)

    # calculate origin tree accuracy
    orig_right_prediction = 0
    for point in useful_valid_data:
        if (int(point[7]) == dt['Left'].get('isLeaf') and point[dt['Attribute'] - 1] <= dt['Value']) \
                or (int(point[7]) == dt['Right'].get('isLeaf') and point[dt['Attribute'] - 1] > dt['Value']):
            orig_right_prediction += 1

    # calculate pruned tree accuracy
    prun_right_prediction = 0
    if dt['Left'].get('Num') >= dt['Right'].get('Num'):
        pruned_label = dt['Left'].get('isLeaf')
    else:
        pruned_label = dt['Right'].get('isLeaf')
    for point in useful_valid_data:
        if point[7] == pruned_label:
            prun_right_prediction += 1

    # print orig_right_prediction, prun_right_prediction

    # find which method has more correct prediction
    if orig_right_prediction >= prun_right_prediction:
        # nothing happend
        return False, None
    else:
        return True, pruned_label

    pass


# if delete
def pruning(valid_data, dt, dir):

    # if itself is leaf
    if dt.get('isLeaf') != 0:
        return dt

    # if itself is not leaf
    else:
        left_child = dt.get('Left')
        right_child = dt.get('Right')

        # record absolute path to root node
        left_dir = dir[:]
        right_dir = dir[:]
        # print left_dir

        left_dir.append([dt['Attribute'], dt['Value'], 'left'])
        # print left_dir
        right_dir.append([dt['Attribute'], dt['Value'], 'right'])

        # left child not leaf
        left_child = pruning(valid_data, left_child, left_dir)
        # right child not leaf
        right_child = pruning(valid_data, right_child, right_dir)

        # after pruning, if both children are leaf, check if delete them
        if left_child.get('isLeaf') != 0 and right_child.get('isLeaf') != 0:
            if_delete, new_Label = check_if_delete_child(valid_data, dt, dir[:])
            # reset this node to delete its children
            if if_delete == True:
                dt['Left'] = None
                dt['Right'] = None
                dt['isLeaf'] = int(new_Label)
                dt['Value'] = 0
                dt['Attrubute'] = 0

        return dt


###########################################PLOTING###########################################


# init
decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def retrieve_tree(i):
    list_of_trees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                    {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                    ]
    return list_of_trees[i]


def cal_leaf_and_depth(dt):

    leaf_num = 0
    depth = 0

    if dt['isLeaf'] != 0:
        return 1, 0
    else:
        if dt['Left'] != None:
            left_leaf, left_depth = cal_leaf_and_depth(dt['Left'])
            leaf_num += left_leaf
        if dt['Right'] != None:
            right_leaf, right_depth = cal_leaf_and_depth(dt['Right'])
            leaf_num += right_leaf

        return leaf_num, max(left_depth, right_depth)



def plot_tree():
    pass


def create_plot():
    fig = plt.figure(1, facecolor='white')





###########################################MAIN STREAM###########################################


conf = np.zeros([4, 4])
conf_pruned = np.zeros([4, 4])

for i in range(10):

    print('\nvalid round: ', i)

    # randomly split data
    train_data, test_data = split_data(noisy_data)
    # train the tree
    dt, depth = decision_tree_learning(train_data, 0)
    this_conf = statistic_data(test_data, dt)

    print('origin accuracy: ', (this_conf[0][0] + this_conf[1][1] + this_conf[2][2] + this_conf[3][3]) / this_conf.sum())
    # use tree to validate and get confusion matrix
    conf += this_conf

    # randomly split again to use another set pruning the tree
    prune_data, test_data = split_data(noisy_data)
    # pruning the whole tree
    pruned_dt = pruning(prune_data, dt, [])
    this_pruned_conf = statistic_data(test_data, pruned_dt)

    print('pruned accuracy: ', (this_pruned_conf[0][0] + this_pruned_conf[1][1] + this_pruned_conf[2][2] +
                                this_pruned_conf[3][3]) / this_pruned_conf.sum())
    conf_pruned += statistic_data(test_data, pruned_dt)


# conf /= 10
print('\norigin confusion mat: \n', conf)
print('\npruned confusion mat: \n', conf_pruned)

print('\norigin accuracy: ', (conf[0][0] + conf[1][1] + conf[2][2] + conf[3][3]) / conf.sum())
print('pruned accuracy: ', (conf_pruned[0][0] + conf_pruned[1][1] + conf_pruned[2][2] + conf_pruned[3][
    3]) / conf_pruned.sum())


###########################################MAIN STREAM###########################################

# print dt

# def visualize(dt):
#     for key in dt.keys():
#         if type(dt[key]).__name__ == 'dict':
#             visualize(dt[key])
#         else:
#             print key, ":", dt[key]
#
# visualize(dt)

#
# desisionNode = dict(boxstyle='sawtooth', fc = "0.8")
# leafNode = dict(boxstyle='round4', fc = '0.8')
# arrow_args = dict(arrow_args = "<-")
# numleafs = 0
# def getNumleafs(decision_tree):
#     global numleafs
#     for key, values in decision_tree.items():
#         if type(values).__name__ == 'dict':
#             getNumleafs(values)
#         elif key =='isLeaf':
#             numleafs+=values
#
# def plotMidText(cntrPt, parentPt, txtString):
#     xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
#     yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
#     creatPlot.ax1.text(xMid, yMid, txtString)
#
# def plotTree(myTree, parentPt, nodeName, numleafs, depth):
#     firstStr = list(myTree.keys())[0]
#     cntrPt = (plotTree.xOff+(0.5/plotTree.totalw+float(numleafs)/2.0/plotTree.totalw), plotTree.yOff)
#     plotMidText(cntrPt, parentPt, nodeName)
#     plotNode(firstStr, cntrPt, parentPt, decisionNode)
#     secondDict = myTree[firstStr]
#     plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
#     for key in secondDict.keys():
#         if type(secondDict[key]).__name__=='dict':
#             plotTree(secondDict[key], cntrPt, str(key))
#         else:
#             plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalw
#             plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
#             plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
#     plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#
# def creatPlot(inTree):
#     fig = plt.figure(1, facecolor='white')
#     fig.clf()
#     axprops = dict(xticks=[], yticks=[])
#     creatPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
#     plotTree.totalw = float(numleafs)
#     plotTree.totalD = float(depth)
#     plotTree.xOff = -0.5/plotTree.totalw
#     plotTree.yOff = 1.0
#     plotTree(inTree, (0.5,1.0), '', numleafs, depth)
#     plt.show()
#
# getNumleafs(dt)
# creatPlot(dt)
