outfile = open('blog_data_short.txt', 'a')

with open("blog_data.txt", 'r', encoding='utf-8') as file:
    count = 0
    while True:
        line = file.readline()
        if not line or count == 1000:
            break

        outfile.write(f"{line}")
        count += 1
