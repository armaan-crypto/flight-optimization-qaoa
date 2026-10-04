rows = set()
toWrite = ""

with open("newscanned.csv", newline="", encoding="utf-8") as f:
    for line in f:
        if "".join(line.split(",")[1:-1]) in rows:
            continue
        toWrite += line
        rows.add("".join(line.split(",")[1:-1]))

with open("removedduplicates.csv", "w", newline="", encoding="utf-8") as f:
    f.write(toWrite)