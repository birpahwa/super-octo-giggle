import openpyxl
import pandas as pd
import xlrd
#wb = openpyxl.load_workbook('islands.xltx')
#print(wb.sheetnames)

#for sheet in wb:
#    print(sheet.title)

#df = pd.read_excel('islands.xltx',sheet_name='Sheet1')
#print(df)

df = pd.read_excel('/Users/birparkash/PycharmProjects/untitled/world bank international arrivals islands v2.xls',sheet_name='Sheet5')
print(df)
