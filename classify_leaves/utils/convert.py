# Generated by ChatGPT 3.5 and modified

import csv
import os.path


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines()]


def write_csv(english_names, chinese_names, output_path):
    with open(output_path, "w", newline="\n", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["英文名", "中文名"])

        for english_name, chinese_name in zip(english_names, chinese_names):
            writer.writerow([english_name, chinese_name])


def main():
    english_file_path = "english_name.txt"
    chinese_file_path = "chinese_name.txt"
    output_csv_path = "../plant_name.csv"

    current_directory = os.path.dirname(os.path.abspath(__file__))
    english_file_path = os.path.join(current_directory, english_file_path)
    chinese_file_path = os.path.join(current_directory, chinese_file_path)
    output_csv_path = os.path.join(current_directory, output_csv_path)

    english_names = read_file(english_file_path)
    chinese_names = [line.split("(")[0].strip() for line in read_file(chinese_file_path)]

    write_csv(english_names, chinese_names, output_csv_path)
    print(f"CSV文件已生成：{output_csv_path}")


if __name__ == "__main__":
    main()
