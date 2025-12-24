"""
测试 Sourcery 代码评审的示例文件
包含多种常见的代码质量问题
"""

import os
import sys
import json
import time


# 问题1: 过于复杂的函数，嵌套过深
def process_user_data(user_data):
    if user_data:
        if 'name' in user_data:
            if user_data['name']:
                if len(user_data['name']) > 0:
                    if user_data['name'].strip():
                        return user_data['name'].strip().upper()
                    else:
                        return None
                else:
                    return None
            else:
                return None
        else:
            return None
    else:
        return None


# 问题2: 重复代码
def calculate_discount_for_vip(price):
    if price > 1000:
        discount = price * 0.2
        final_price = price - discount
        tax = final_price * 0.1
        return final_price + tax
    else:
        discount = price * 0.1
        final_price = price - discount
        tax = final_price * 0.1
        return final_price + tax


def calculate_discount_for_regular(price):
    if price > 1000:
        discount = price * 0.1
        final_price = price - discount
        tax = final_price * 0.1
        return final_price + tax
    else:
        discount = price * 0.05
        final_price = price - discount
        tax = final_price * 0.1
        return final_price + tax


# 问题3: 不必要的 else 语句
def check_age(age):
    if age >= 18:
        return "Adult"
    else:
        return "Minor"


# 问题4: 可以使用列表推导式
def get_even_numbers(numbers):
    result = []
    for num in numbers:
        if num % 2 == 0:
            result.append(num)
    return result


# 问题5: 低效的字符串拼接
def build_message(items):
    message = ""
    for item in items:
        message = message + str(item) + ", "
    return message


# 问题6: 使用可变默认参数
def add_item(item, item_list=[]):
    item_list.append(item)
    return item_list


# 问题7: 过长的函数参数列表
def create_user(name, email, age, address, phone, city, country, zipcode, occupation, company):
    return {
        'name': name,
        'email': email,
        'age': age,
        'address': address,
        'phone': phone,
        'city': city,
        'country': country,
        'zipcode': zipcode,
        'occupation': occupation,
        'company': company
    }


# 问题8: 不必要的变量赋值
def calculate_total(a, b):
    result = a + b
    return result


# 问题9: 可以使用 with 语句
def read_file(filename):
    f = open(filename, 'r')
    content = f.read()
    f.close()
    return content


# 问题10: 复杂的布尔表达式
def is_valid_user(user):
    if user is not None and user.get('active') == True and user.get('verified') == True and user.get('banned') == False:
        return True
    return False


# 问题11: 使用 type() 而不是 isinstance()
def check_type(value):
    if type(value) == str:
        return "string"
    elif type(value) == int:
        return "integer"
    else:
        return "other"


# 问题12: 空的 except 块
def risky_operation():
    try:
        result = 10 / 0
        return result
    except:
        pass


# 问题13: 不必要的 lambda
def sort_users(users):
    return sorted(users, key=lambda x: x['name'])


# 问题14: 可以使用字典的 get 方法
def get_user_name(user_dict):
    if 'name' in user_dict:
        return user_dict['name']
    else:
        return 'Unknown'


# 问题15: 重复的条件判断
def categorize_score(score):
    if score >= 90:
        category = "A"
    if score >= 80 and score < 90:
        category = "B"
    if score >= 70 and score < 80:
        category = "C"
    if score < 70:
        category = "F"
    return category


class UserManager:
    # 问题16: 方法可以是静态方法
    def validate_email(self, email):
        return '@' in email and '.' in email
    
    # 问题17: 过于复杂的方法
    def process_users(self, users):
        valid_users = []
        for user in users:
            if user:
                if 'email' in user:
                    if self.validate_email(user['email']):
                        if 'age' in user:
                            if user['age'] >= 18:
                                valid_users.append(user)
        return valid_users


# 问题18: 未使用的变量
def calculate_something():
    x = 10
    y = 20
    z = 30
    return x + y


if __name__ == "__main__":
    # 测试代码
    print(process_user_data({'name': '  test user  '}))
    print(get_even_numbers([1, 2, 3, 4, 5, 6]))
    print(check_age(25))
