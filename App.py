import os.path
from Detect import detect

rootdir = ".\\TestImage"
for each in range(0,8):
    tem = rootdir + "\\" + str(each)
    print(tem)
    detect(tem)


# ######################命令行程序###########################
# iterations = 10
# if __name__ == '__main__':
#     flag = True
#     while(flag):
#         print("1:检测   "   + "2:参数设置   " + "3:退出 ""\n")
#         select = input("请选择：")
#         if select == "1":
#             rootdir = input("请输入要处理的文件路径：")
#             if os.path.exists(rootdir):
#                 print("请稍后...." + "\n")
#                 detect(rootdir,iterations)
#                 print("完成" + "\n")
#             else:
#                 print("无此路径" + "\n")
#
#         elif select == "2":
#             print("1:分割迭代次数   " + "2:无   " + "3: 无""\n")
#             select = input("请选择：")
#             if select == "1":
#                 print("当前值" + str(iterations)+ "\n")
#                 inpu= input("请输入设置值：""\n")
#                 iterations = int(inpu)
#             else:
#                 print("无设置")
#
#         elif select == "3":
#             flag = False
#
#         else:
#             print("请输入正确选项" + "\n")
