import pandas as pd

url = "/Users/tony/账户信息需再次提交的人员名单.xls"

sheet1 = pd.read_excel(url)

charge = 0

for i in sheet1['姓名']:
    if i == '杨涵':
        charge = 1
        print(i)

if charge:
    print("WTF")
else:
    print("OK")





