# import random, string

# def passwordGenerator(length= 10):
#     character= string.ascii_letters + string.digits + string.punctuation
    
#     password = "".join((random.choice(character)) for i in range (length))
#     return password
   

# print(passwordGenerator(50))
# def online_count(status):
#     online_counts = 0
#     status= {
#        "Alice": "online",
#     "Bob": "offline",
#     "Eve": "online",
#     }
#     # for i in status:
#     #     if status{len(status)}== "online":
#     #         online_counts += 1
          
#     return online_counts

# print(online_count())
# def double_letters(word):
#     for i in range(len(list(word))):
#         for i in list(word) :
#             if i ==:
#                print(i)
#                i-= 1
#         #     return True
#         # else :
#         #     return False
# double_letters("Hello")


# def add_dots(stirng):
#     group= list(stirng)
#     dottedString= ".".join(group)
#     for i in range(len(group)):
#         return dottedString
# print(add_dots("helkjsfkkljjjjlk55klaflo"))

# def remove_dots(string):

#     return str(string).replace(".", "")
# print(remove_dots(add_dots("helkjsfkkljjjjlk55klaflo")))


# def count(string):
#     count= str(string).count("-")
#     return count+1
# print(count("ho-te-l"))

# import random

# def is_anagram(string1, string2):
#     random.shuffle(list(string1))
#     random.shuffle(list(string2))
#     # if (str(string1).count== str(string2).count) and (list(string1)==list(string2)) :
#     print(string1)
#     print(list(string1), list(string2))
#     return True
#     return False
# print(is_anagram("hello", ("helol")))


# def flatten(list):
#     flatten_list= []
#     for row in list:
#         flatten_list.extend(row)
#     return flatten_list
# print(flatten([["fesef", "fsfsfs"],["fjls", "kfksk"]]))




# def largest_difference(numbers):
#     list(numbers)
#     result= max(numbers)- min(numbers)
#     return result
# print(largest_difference([5, 6, 10]))        
    
    
# def dev_3(number):
#     if number %3 == 0:
#         return True
#     return False
# print(dev_3(30))


# import string
# def palindrome(s:str):
#     if len(s) > 0:
#         accepted_characters =string.ascii_lowercase+string.digits
#         list1 = []
#         list2 = []
#         for i in s:
#             if i in accepted_characters:
#                 list1.append(i)
#         for i in s[::-1].lower():
#             if i in accepted_characters:
#                 list2.append(i)
#         return list1 == list2, list1, list2
#     return True
            

# print(palindrome("    sfs"))


# def up_down(number):    
#     upNumber =number+1
#     downNumber= number- 1    
#     return upNumber, downNumber
# print(up_down(3))


# import re
# def consecutive_zeros (s):
#     matches = re.findall(r'((\w)\2{1,})', s)
#     list = [match[0] for match in matches if match[1]== "0"]
#     length = [len(i) for i in list]
#     return (max(length))
# print(consecutive_zeros ("1001101000000100011100"))
 


# def all_equals(list):
#    for i in list :
#     if all(item == list[0] for item in list ):
#         return True
#    return False
# print(all_equals(["hehlw", "hehlw", "hehlw"]))
# def dis(price, discount):
#     return round(float(price-( price* discount/100)))
# print(dis(911, 400))

# def calculator(num1, operator, num2):
#     if operator == "+":
#         resut = num1 +  num2 
#     elif operator == "-":
#         resut = num1 -  num2  
#     if operator == "*":
#         resut = num1 *  num2 	 
#     if operator == "/":
#         resut=  num1 /  num2  	 
			 
		
#     return 	resut
			 
# print(calculator(50, "*", 20))


# import string 
# def isFound(char):
#      for letter in list(string.ascii_letters):
#          if letter == char:
#              return True
#      return False

# print(isFound("f"))      

# def color_invert(r, g, b ):
   
#     numbers= (list(range(0, 256)))
#     join= "".join(numbers)
#     reversedNumbers= "".join(numbers[::-1])
    
#     translation_table = str.maketrans(join , reversedNumbers)
    
#     color = (r.translate(translation_table) , g.translate(translation_table), b.translate(translation_table))
    
#     return  color
# print(color_invert(10, 0, 5))

# from math import ceil
# def end_corona(recovers, new_cases, active_cases):
#     return ceil(float((active_cases)/(recovers-new_cases)))

# print(end_corona(4000, 2000, 77000))

# print (alphabet)


# numbers= (list(range(0, 256)))
# print(slice(x))
# reversedNumbers= numbers[::-1]
# print(reversedNumbers)










# def encode_morse(message):
#     messages= list(message.upper())
#     char_to_dots = {
#   'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
#   'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
#   'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
#   'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
#   'Y': '-.--', 'Z': '--..', ' ': ' ', '0': '-----',
#   '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
#   '6': '-....', '7': '--...', '8': '---..', '9': '----.',
#   '&': '.-...', "'": '.----.', '@': '.--.-.', ')': '-.--.-', '(': '-.--.',
#   ':': '---...', ',': '--..--', '=': '-...-', '!': '-.-.--', '.': '.-.-.-',
#   '-': '-....-', '+': '.-.-.', '"': '.-..-.', '?': '..--..', '/': '-..-.'}

#     a= list(char_to_dots.keys())
#     b = list(char_to_dots.values())
#     i = 0
#     for letter in messages:
#         count= a.index(letter)
#         messages[i]= b[count]
#         i+= 1
#     a = " ".join(messages)
#     return str(a)
# print(encode_morse("Hello I am Suhaib"))













# ahmed = "ahmed"
# list= list(ahmed)
# for letter in list:
#     if letter == "a":
#        a= list.index(letter)
#        list[a]= "v"   

# print(list, letter, a)




# import math 
# def radians_to_degrees(rad):
#     radians = 180 /math.pi 
#     return rad*radians
    
# print(radians_to_degrees(1))






# def convert(numbers):
#     return [str(number) for number in numbers]
        
# print(convert([1, 2, 3]))





# def validate(code):
#     for i in code.split():
#         if i =="def":
#             return True
#         else :
#             print ("missing def ")
#             if i !=":":
#                return False
#             else :
#                 print ("missing : ")
#                 if i =="validate":
#                      return True
#                 else :
#                   print ("wrong name")
#                   if i =="(" and i== ")":
#                     return True
#                   else :
#                     print ("missing parameter ")
#                     if i =="return":
#                         return True
#                     else :
#                        print ("missing return ")
#                        if i !="()":
#                           return True
#                        else :
#                           print ("missing parameter")
#                           if i =="    ":
#                              return True
#                           else :
#                             print ("missing return ")
    
    
    
#     return code.split() 
# print(validate("def validate(code):"))
    



# import collections
# def list_xor(n, list1, list2):
#     list1.extend(list2)
#     repeated= collections.Counter(list1)
#     # for i in [repeated.values()]:
#     #     if i > 1:
#     #         return True 
#     return list1 , repeated.items() , collections.Counter(list1)
# print(list_xor(1, [0, 2, 3], [1, 5, 6])) 





# my_list = ["a", "b", "a", "c", "c", "a", "c"]
# counts = collections.Counter(my_list)
# print("Repeated items:")
# for item, count in counts.items():
#     if count > 1:
#         print(f"{item}: {count}")
#         print(counts.items())









# def param_count(*args):
    
#     return len(args)
 
# print(param_count(1, 2, "apple", "hello"))




# def format_number(number):
#     s = list(str(number))
#     r= 0
#     p = len(s)
#     for i in range(0,p):
#         r+= 1
#         p+= 1
#         if r %4==0 and p!= 4:
#            s.insert(r,"," )
#     if p>1: 
#        s.insert(1, ",")
#     pro = "".join(s)    
#     p = len(s)
#     return  pro
# print(format_number(100000000))
# ahmed= 1000
# a = (f"{ahmed:}")
# print( a,type(ahmed), type(a))


# def format_number(number):
#     return "{:,}".format(number)
# print(format_number(1000000))


# number = ["1"," 2"," 3", "4", "5", "6"]
# j= number[::-1]
# j.remove("5")
# print("".join(j))

# import collections 
# def advanced_sort(lists):
#     repeated = "".join(list(lists))
#     i = 0
#     for item in repeated:
#        if repeated.index(item) == repeated.index(item)-1:
#           repeated.index(item)
#           i+= 1
#           return item 
         
    


# print(advanced_sort([1, 2, 3, 5, 1, 1, 6, 1]))


# def find():
#     ahmed= "ahhmed"
#     for h in ahmed :
#         if h == "a":
#             return [ahmed.index(h)]

# print(find())









# import datetime
# def friday(month, year):
#     date = datetime.datetime(year, month, 26)
    
#     if date.weekday()== 6 :
#         return True
#     return False
    
# print(friday(12, 2004))




# import collections
# def difference_letter(word1, word2):
#     lists = list(word1) + list(word2)
#     s= collections.Counter(lists)
#     for letter, count in s.items():
#         if count <2:
#             return f"'{letter}' is the unique letter in the list"
    

# print(difference_letter("high", "highs"))


# def average(*args):
#     return (sum((args),0))/2
# print(average(10))



# import string
# def check(word):
#     new= "" 
#     for letter in list(word):
#         if letter.islower():
#            new+= letter.upper()
#         elif letter.isupper():
#             new+= letter.lower()
#     return word, new, letter
# print(check("hI"))




# def deleteIng(string):
#     if string[-3:] == "ing" or string[-3:] =="ING":
#        return f"to {string[:-3]} :) "
#     return f"{string} has no 'ing' in the end"
# print(deleteIng("accounting"))

# def commify(list1):
#    new_list= ", ".join(list1)
#    new  = list(new_list)
#    new[ new_list.rindex(",")]= "and"
#    new.insert( new_list.rindex(","), " ")
   
#    return "".join(new)
# print(commify(["hello", "high", "hi", "welcome to the united states bro"]))





# ahmed= list("building")
# s= ahmed[::-1]
# s.remove(s[0]) , s.remove(s[1]), s.remove(s[2])
# new_word= "".join(s[::-1])
# print(ahmed, s) 

# ahmed= "hello"
# s= ahmed.replace("h", "")
# p = s.replace("e", "")
# print(s, p)








# def there(list):
#    if list[0]== 1 or list[0]==0:
#       return f"There is {list[0]} {list[1]}"
#    else :
#       return f"There are {list[0]} {list[1]}s"
# print(there([1, "brother"]))












# import turtle
# myTurtle= turtle.Turtle()
# myTurtle.speed(1)
# myTurtle.color("red")
# def draw_circle(length, angle):
#    for i in range(1):
#        turtle.circle(50)
#        turtle.left(70)
#        turtle.forward(50)
#        turtle.circle(30)
# draw_circle(70, 70 )



# import random

# def gussingGame():   
#     try:   
#         print("Hello, and welcome to the Quiz Game :)")
#         name= input("please enter your name: ").capitalize()
#         age = int(input("Please enter your age : "))
#         if age >= 18 :
#                     print(f"Hello, {name}, you are allowed to play this game ")
#         else: 
#                     print(f"Sorry! {name},  you are not allowed to play this game")
#                     quit()
#         trials= 0
#         score = 0
#         guesses= int(input("How many times do you want to guess ? "))
#         playing = input("Do you want to play (yes, no): ").lower()
#         randomNumber = random.randrange(10, 150)
#         running= True
#         while running:    
#             if trials == guesses:
                            
#                             print(f" You are out of trials! Your score is {score} , the number was {randomNumber}")
#                             break
#             if playing== "yes":
            
#                     numberInput= int(input("Guess a number between 10 , 150 : "))
#                     if numberInput == randomNumber:
#                             score+=1
#                             trials+=1
#                             print(f"Correct! You are great :) You got it in {trials} trials")
#                     elif numberInput> randomNumber:
#                         print("You are above the number")
#                         trials+= 1
#                     elif numberInput< randomNumber:
#                         print("You are under the number")       
#                         trials+= 1
                    
#                     # else:
#                     #         print("Ops! This is wrong")
#                     #         trials+=1
#                     #         continue
                    
                    
#             elif playing== "no":
#                 print("Thanks for choosing Quiz Game :)")
#                 quit()
#             else: 
#                 print("Wrong input, Try Again")
#                 continue        
#     except:
#         print("wrong input")
#         running = False



# gussingGame()


# from cryptography.fernet import Fernet
# def read_key():
#     file = open ("key.key", "rb")
#     key = file.read()
#     file.close()
#     return(key)


# master= input("What is the password manager passkey : ")
# key = read_key() + master.encode()
# fer = Fernet(key)

         
# def delete():
#     print()
# def add():
#     name= input("Account : ")
#     password= input("Password : ")
#     with open("passmanager.txt", "a") as f:
#         f.write(f"{name} : {str(fer.encrypt(password.encode().decode()))}\n")

# def view():
#     check = input("What is your key ? ")
#     with open("passmanager.txt", "r") as f:
#          for line in f.readlines():
#              data = line.rstrip()
#              user , passw = data.split(":")
#          if check == read_key():
#                 print(f"Username is {user}, Password is {str(fer.decrypt(passw.encode()))}")
#          else: print("The key you entered is wrong :)")



# running = True
# trials= 5
# while True and trials > 0:
#     if master == "sohaib":
#         running== True
#     else:
#         trials-= 1
#         print(f"Wrong Password , {trials} trials left")
#         continue
#     user_input=  input("Would you like to add a password or view or delete an existing one (add, view, delete or quit): ").lower()
#     if user_input == "view":
#         view()
#     elif user_input == "add":
#         add()
#     elif user_input== "delete":
#         delete()
#     elif user_input== "quit":
#         quit()
#     else :
#         print("Invalid Mode")
#         continue

# def findPosition(car , gas):
#         number =   int( (len(gas))  /2)
#         i = 0
#         while True:
#             i += 1
#             if car == gas[number -1]:
#                 print ("hi")
#                 break
#             elif car > gas [number -1]:
#                 number+1 *2 
#             elif car < gas[number -1]:
#                 number+1  /2
#             else : print ("Something is wrong")

# findPosition(2, [1 , 2 , 3, 4, 5, 6 , 7, 8, 9, 10])

# def insertion(arr):
#     for i in range (1, len(arr)):
#         j = i 
#         while arr[j- 1]> arr[j] and j > 0:
#             arr[j- 1],arr[j]  =arr[j] , arr[j-1]
#             j-= 1
#     print (arr)
# insertion([1, 5, 6, 3 , -1 , 6, -10, 0])




# def selection_minimum(arr):
#     for i in range (1, len(arr)):
#         cur_min = arr[0]
#         j = i 
#         while cur_min > arr[j] and j > 0:    
#             cur_min = arr[j]
#             j-= 1
#     print (cur_min)
# selection_minimum([10, 5, 6, -5 , 0 , -10, 6, 6, 0])







# def mid(string):
#     listt = [i for i in string]
#     if len(listt)%2 != 0:
#         return listt[len(listt)//2]
#     else:
#         return '""'
    
# print(mid("hell"))
    

# def count(string):
#     return len(string.split("-"))

# print(count("Hel-l"))




# def div_3(parameter):
#     if (parameter%3)<=0:
#        return True
#     return False 
# print(div_3(65))

# def up_down(number):
#     return (number-1 , number+1)
# print(up_down(5))

# def flatten(list):
#     big_list = []
#     for i in range(len(list)):
#         big_list.extend(list[i])
#     return big_list
# print(flatten([[1 , 2] , [5 , 6] , [56]]))
# def int_to_str(int):
#     return f"{int}"
# print(int_to_str(5))

# def str_to_int(str):
#     return int(str)
# print(type(int_to_str(5)) , type(str_to_int("3")))

# import cs50
# from sys import argv , exit
# import cowsay

# name = cs50.get_string("Enter a string: ")
# print(f"{name} is your name :)")
# print(f"lower case of your name is: {name.lower()}")
# name = " hello"
# if len(argv) != 2:
#     print("Missing command-line argument!")
#     exit()
# else:
#     print(f"Hello, {argv[1]} :)")

# def isSpace(string):
#     spaces = 0
#     for space in string:
#         if space == " ":
#             spaces+=1 
#     if spaces == len(string):
#         return True
#     return False

# print(isSpace(name))
# print(name.isspace())
# n = 1000
# list = [i for i in range(n+1)]
# for _ in list:
#     print(f"The world is the best in the {list[_]}")
    
# import csv
# from cs50 import SQL
# db = SQL("sqlite:///data.csv")
# favorite = input("Favorite: ")

# rows = db.execute("SELECT COUNT(*) AS n FROM fovorites WHERE problem = ?", favorite)

# row = rows[0]
# print(row["n"])



# class Solution:
#     def fizzBuzz(self, n: int)-> List[str]:
#         output= []
#         for i in range(1, n+1):
#             if (i %3== 0) and (i %5==0):
#                 output.append(("FizzBuzz"))
#             elif i %3==0 :
#                 output.append(("Fizz"))
#             elif i %5 == 0 :
#                 output.append(( "Buzz"))
#             else:
#                 output.append(str(i))
#         return list(output)




# class Solution:
#     def isPowerOfThree(self, n: int) -> bool:
#         if n < 1:
#             return False
#         while n % 3 == 0:
#             n //= 3
#         return n == 1
    


# class Solution:
#     def romanToInt(self, s: str) -> int:
#         # Dictionary to store Roman numeral values
#         roman_values = {
#             'I': 1,
#             'V': 5,
#             'X': 10,
#             'L': 50,
#             'C': 100,
#             'D': 500,
#             'M': 1000
#         }
        
#         total = 0
#         prev_value = 0
        
#         # Iterate over each character in the string
#         for char in reversed(s):
#             value = roman_values[char]
#             # If the current value is less than the previous, subtract it (e.g., IV = 4)
#             if value < prev_value:
#                 total -= value
#             else:
#                 total += value
#             prev_value = value
        
#         return total


# def firstUniqChar(s: str) -> int:
#     # Step 1: Create a frequency array for 26 lowercase letters
#     occurance = [0] * 26
    
#     # Step 2: Count occurrences of each character
#     for char in s:
#         occurance[ord(char) - ord('a')] += 1
    
#     # Step 3: Find the first unique character by checking counts
#     for index, char in enumerate(s):
#         if occurance[ord(char) - ord('a')] == 1:
#             return index
    
#     return -1

# print(firstUniqChar("hfghfhgellowworld"))

# def is_anagram(t:str,s: str):
#     list1= list(s)
#     list2 = list(t)
#     list1.sort()
#     list2.sort()
#     return list2 == list1

# #     occurance = [0] * 26
    
# #     for char in t:
# #         occurance[ord(char) - ord('a')] += 1
# #     for char in s:
# #         occurance[ord(char) - ord('a')] += 1
        
# #     for _ in occurance:
# #         if _ %2 !=0:
# #             return False
# #     return True

# print(is_anagram("abba","abab"))

# from collections import Counter

# s = "bba"
# t = "abb"
# print(Counter(s)==Counter(t))
# print(Counter(t),Counter(s))

# from textblob import TextBlob        *****Parse language*****

# def analyze_arabic_sentiment(text):
#     """Analyzes the sentiment of an Arabic sentence.

#     Args:
#         text (str): The Arabic sentence to analyze.

#     Returns:
#         float: A sentiment score between -1 (very negative) and 1 (very positive).
#     """

#     blob = TextBlob(text)
#     sentiment = blob.sentiment.polarity
#     return sentiment

# # Example usage:
# arabic_sentence = "وين الجديد ل و سمحتم يااستاذ محمد"  # "I love reading"
# sentiment_score = analyze_arabic_sentiment(arabic_sentence)

# if sentiment_score > 0:
#     print("Positive sentiment")
# elif sentiment_score < 0:
#     print("Negative sentiment")
# else:
#     print("Neutral sentiment")







# # **** Gemeini Sentiment Code **** -> Doesn't work

# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from arabic_stopwords import stopwords
# from nltk.stem.isw import ISWStemmer

# # Load the pre-trained model and tokenizer
# model_name = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# # Preprocessing function
# def preprocess(text):
#     # Tokenization
#     tokens = tokenizer.tokenize(text)

#     # Handle diacritics (if necessary)
#     # You can use libraries like ArabicStemmer or ArabicLightStemmer for diacritic handling

#     # Normalize text (if needed)
#     # Consider using normalization techniques like stemming or lemmatization

#     # Remove stop words
#     tokens = [token for token in tokens if token not in stopwords]

#     # Stemming
#     stemmer = ISWStemmer()
#     tokens = [stemmer.stem(token) for token in tokens]

#     # Convert tokens to input IDs
#     input_ids = tokenizer.convert_tokens_to_ids(tokens)

#     # Pad or truncate input_ids to a fixed length
#     max_length = 128  # Adjust as needed
#     input_ids = input_ids[:max_length]
#     input_ids = input_ids + [0] * (max_length - len(input_ids))

#     # Create attention mask
#     attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))

#     return torch.tensor([input_ids]), torch.tensor([attention_mask])

# # Sentiment analysis function
# def predict_sentiment(text):
#     inputs = preprocess(text)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         predicted_class = torch.argmax(logits, dim=1).item()

#     sentiment_labels = ['negative', 'neutral', 'positive']
#     predicted_sentiment = sentiment_labels[predicted_class]

#     return predicted_sentiment

# # Example usage
# text = "هذا مقال رائع جداً"  # Arabic sentence: "This is a very great article"
# predicted_sentiment = predict_sentiment(text)
# print(predicted_sentiment)  # Output: 'positive'

# Improving accuracy:
# - Experiment with different pre-trained models (e.g., aubmindlab/bert-base-arabertv02)
# - Fine-tune the model on a larger, more diverse Arabic sentiment analysis dataset
# - Consider using data augmentation techniques to increase training data diversity
# - Explore advanced preprocessing techniques like word embeddings or character-level features
# - Optimize hyperparameters like learning rate, batch size, and number of epochs
# - Evaluate the model's performance using appropriate metrics (e.g., accuracy, precision, recall, F1-score)






# **** ChatGPT Sentiment Code **** -> Great


# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from transformers import pipeline
# import torch
# import re

# # Load the tokenizer and model (you can choose either 'aubmindlab/bert-base-arabertv02' or 'CAMeL-Lab/bert-base-arabic-camelbert-mix')
# model_name = "aubmindlab/bert-base-arabertv02"  # or "CAMeL-Lab/bert-base-arabic-camelbert-mix"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 for 'positive', 'neutral', 'negative'

# # Function to remove Arabic diacritics
# def remove_diacritics(text):
#     arabic_diacritics = re.compile(r'[\u064B-\u065F]')
#     return re.sub(arabic_diacritics, '', text)

# # Normalize Arabic text
# def normalize_arabic(text):
#     text = re.sub("[إأآا]", "ا", text)
#     text = re.sub("ى", "ي", text)
#     text = re.sub("ؤ", "و", text)
#     text = re.sub("ئ", "ي", text)
#     text = re.sub("ة", "ه", text)
#     return text

# # Preprocess the input sentence (diacritic removal and normalization)
# def preprocess_text(text):
#     text = remove_diacritics(text)
#     text = normalize_arabic(text)
#     return text

# # Define a sentiment analysis pipeline
# def analyze_sentiment(text):
#     preprocessed_text = preprocess_text(text)
#     inputs = tokenizer(preprocessed_text, return_tensors='pt', truncation=True, padding=True)
#     outputs = model(**inputs)
#     probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     sentiment = torch.argmax(probs).item()

#     # Map the sentiment to 'positive', 'neutral', 'negative'
#     sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
#     return sentiment_mapping[sentiment] # type: ignore

# # Example usage
# text = "و لا تحسبن الذين قتلوا في سبيل الله أمواتا بل احياء عند ربهم يرزقون"
# sentiment = analyze_sentiment(text)
# print(f"Sentiment: {sentiment}")





# **** Copilot Sentiment Code **** -> Error


# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# def preprocess_text(text):
#     # Normalize text
#     text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
#     text = text.replace("ة", "ه")
#     text = text.replace("ى", "ي")

#     # Remove diacritics
#     diacritics = ["َ", "ً", "ُ", "ٌ", "ِ", "ٍ", "ْ", "ّ"]
#     for diacritic in diacritics:
#         text = text.replace(diacritic, "")
    
#     return text

# # Load pre-trained model and tokenizer
# model_name = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# def predict_sentiment(text):
#     # Preprocess the text
#     text = preprocess_text(text)
    
#     # Tokenize the text
#     inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
#     # Perform the prediction
#     outputs = model(**inputs)
#     predictions = torch.softmax(outputs.logits, dim=1)
#     sentiment = torch.argmax(predictions).item()

#     if sentiment == 0:
#         return 'negative'
#     elif sentiment == 1:
#         return 'neutral'
#     else:
#         return 'positive'

# # Test the function
# text = "والله اشتقنا لك اخي محمد 2024"
# sentiment = predict_sentiment(text)
# print(f"Sentiment: {sentiment}")








# def longestCommonPrefix(strs: list) -> str:
#     if not strs:
#         return ""
    
#     # Start with the first string as a potential prefix
#     prefix = strs[0]
    
#     # Compare the prefix with each string in the list
#     for word in strs[1:]:
#         # Update the prefix by checking common characters from the start
#         while word[:len(prefix)] != prefix:
#             # Reduce the prefix by removing the last character
#             prefix = prefix[:-1]
#             if not prefix:
#                 return ""
    
#     return prefix


# # Test case
# print(longestCommonPrefix(["flowser", "flows", "lf"]))




# def strStr(haystack: str, needle: str) -> int:
#         if needle in haystack:
#             return haystack.index(needle)
#         else:
#             return -1
        
        
    
    
    
    
    
    
# print(strStr("mississippi","issip"))
    
    
    
    
    
    
    
    
# def reverse(x: int) -> int:
#     # Define the 32-bit integer limits
#     INT_MAX = 2**31 - 1
    
#     # Handle negative numbers by storing the sign
#     sign = -1 if x < 0 else 1
#     x = abs(x)
    
#     # Reverse the digits
#     reversed_x = 0
#     while x != 0:
#         # Extract the last digit and remove it from x
#         digit = x % 10
#         x //= 10
        
#         # Check for overflow before adding the digit
#         if (reversed_x > INT_MAX // 10) or (reversed_x == INT_MAX // 10 and digit > INT_MAX % 10):
#             return 0
        
#         # Add the digit to the reversed number
#         reversed_x = reversed_x * 10 + digit
    
#     return sign * reversed_x

# print(reverse(-995))



# def countVowls(string:str):
#     vowls = ["a", "o", "u", "e", "i"]
#     count = 0
#     for char in string.lower():
#         if char in vowls:
#             count+=1       
#     return count
# print(countVowls("Hello wOrld"))



# def removeDuplicates(self, nums: list[int]) -> int:
#     if not nums:
#         return 0

#     # Initialize the first pointer
#     i = 0

#     # Use the second pointer to iterate through the list
#     for j in range(1, len(nums)):
#         if nums[j] != nums[i]:
#             i += 1
#             nums[i] = nums[j]

#     # The length of unique elements is i + 1
#     return i + 1


# nums = [7,1,5,3,6,0,4]


# min = min(nums)

# print(min,nums.index(min))




# def reverse(list1):
#     reversed = []
#     for element in list1[::-1]:
#         reversed.append(element)
#     return reversed

# print(reverse([7,1,5,3,6,0,4,8]))


# def multiplied_by_item():
#     num = int(input("Enter a number: "))
#     factor = int(input("Enter how many numbers to multiply by: "))
#     factors = [i for i in range(factor+1)]
    
#     for result in factors:
#         print(f"{num} * {result} = {num*result}")
        
        
        
# multiplied_by_item()





# import random
# list = [i for i in range(7)]
# random.shuffle(list)
# print(list)
# def maxProfit(prices:list):
#     profit = 0
#     for index in range(len(prices)-1):
#         current_index = index+1
#         if prices[index] < prices[current_index]:
#             profit += prices[current_index] - prices[index]
#         else:
#             continue
            
#     return profit
    
    
# print(maxProfit(list))


# def rotate(nums: list[int], k: int)->None:
#     k = k%len(nums)
#     nums[:] = nums[-k:]+nums[:-k]
# print(rotate([1,2,3,4,5,6,7],3))

# from collections import Counter
# def containsDuplicate(nums: list[int]) -> bool:
#     for duplicate in Counter(nums).values():
#         if duplicate > 1:
#             return True
#     return False

# print(containsDuplicate([1,2]))



# def singleNumber(nums: list[int]) -> int:
#     numbers = [0]*10
#     for _ in nums:
#         numbers[ord(f"{_}")-ord("0")]+=1
#     return numbers.index(1)
# print(singleNumber([2,2,1,1,5]))



# def intersect(nums1: list[int], nums2: list[int]) -> list[int]:
#     intersect = []
#     for num1 in nums1:
#         for num2 in nums2:
#             if num1 == num2:
#                 intersect.append(num1)
#                 del(nums2[nums2.index(num2)])
#                 break
                
#     return intersect


# print(intersect([4,9,5],[9,4,9,8,4]))


# nums1 = [1,2,3,4,4]
# count = {}
# for num in nums1:
#             count[num] = count.get(num,0)+1
            
            
# print(count)


# num = [9]
# string = ""
# for _ in range(len(num)):
#     string+=str(num[_])


# print(str(int(string)+1))



# def plusOne(digits: list[int]) -> list[int]:
#     number = ""
#     for digit in digits:
#         number+=str(digit)
#     number = int(number)+ 1
#     return [int(x) for x in str(number)]
# print(plusOne([9,9]))






# def moveZeroes(nums: list[int]) -> None:
#     for i in range(len(nums)):
#         print(i, nums[i])
#         if nums[i] == 0:
#             nums.append(nums[i])
#             nums.remove(nums[i])
#     return nums

# print(moveZeroes([0,0,1]))





# def twoSum(nums: list[int], target: int) -> list[int]:
#     if len(nums) < 2:
#         return False
#     for num in nums:
#         for _ in nums[nums.index(num)+1:]:
#             if num + _ == target:
#                 return([nums.index(num),nums.index(_)])
# print(twoSum(nums = [3,3], target = 6))


# def twoSum(nums: list[int], target: int) -> list[int]:
#     num_to_index = {}
#     for i, num in enumerate(nums):
#         complement = target - num
#         if complement in num_to_index:
#             return [num_to_index[complement], i]
#         num_to_index[num] = i
#     return []

# # Test the function
# print(twoSum(nums=[3, 3], target=6))





# def rotate(matrix: list[list[int]]) -> None:
    
#     return matrix[:]
    
    
# print(rotate([[1,2,3],
#               [4,5,6],
#               [7,8,9]]))


# i = int(input("Enter the times you want to repeat: "))
# running = 0
# while running < i:
#     name = input("What is your name? ")
#     print(f"Hello, {name}")
#     running+=1
    
# import itertools
# import string

# def generate_passwords(output_file, length):
#     characters = string.ascii_lowercase + string.digits
#     with open(output_file, 'w') as f:
#         for password in itertools.product(characters, repeat=length):
#             f.write(''.join(password) + '\n')

# if __name__ == "__main__":
#     length = 3  # Password length
#     output_file = "passwords.txt"
#     generate_passwords(output_file, length)
#     print(f"Password file '{output_file}' generated.")

