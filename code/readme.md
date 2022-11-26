AML.ipynb为jupyter上的整个代码流程，可以按照我的注释一步一步的复现
AML.py是将AML.ipynb的代码放入py文件中了，建议执行上面的ipynb文件更加清楚
注意，数据需要新建一个data文件夹，然后在data文件夹新建aml文件夹，将数据集存放在aml文件夹内即可
如果只有一块GPU，则将代码中的cuda:1全部改为cuda:0

库要求：
torch版本1.11
python版本3.7
tqdm
pytorch——Bertpretrained等