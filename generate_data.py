import json, random, os

num_words = {
    "0":"zero","1":"one","2":"two","3":"three","4":"four",
    "5":"five","6":"six","7":"seven","8":"eight","9":"nine"
}

def spell(n):
    return " ".join(num_words[d] for d in n)

def gen_phone():
    d = "".join(str(random.randint(0,9)) for _ in range(10))
    return spell(d), d, "PHONE"

def gen_credit():
    d = "".join(str(random.randint(0,9)) for _ in range(16))
    return spell(d), d, "CREDIT_CARD"

def gen_email():
    name = random.choice(["alice","rahul","maria","neha","amit"])
    dom = random.choice(["gmail","yahoo","hotmail"])
    spoken = f"{name} dot work at {dom} dot com"
    return spoken, f"{name}.work@{dom}.com", "EMAIL"

def gen_person():
    f = random.choice(["rahul","neha","sara","amit","john"])
    l = random.choice(["kumar","sharma","roy","patel"])
    return f"{f} {l}", f"{f} {l}", "PERSON_NAME"

def gen_city():
    c = random.choice(["delhi","mumbai","london","paris"])
    return c, c, "CITY"

def gen_location():
    loc = random.choice(["railway station","bus stop","city mall"])
    return loc, loc, "LOCATION"

def gen_date():
    d = random.choice(["first jan","third march","twenty first august"])
    return d, d, "DATE"

gens = [gen_phone, gen_credit, gen_email, gen_person, gen_city, gen_location, gen_date]

temps = [
    "Here is the __ENTITY__ you requested.",
    "Please record the __ENTITY__ now.",
    "This is the __ENTITY__ from the customer.",
    "You asked for the __ENTITY__, here it is.",
    "The user mentioned __ENTITY__ on the call."
]

def build():
    fn = random.choice(gens)
    ent, real, lab = fn()
    temp = random.choice(temps)
    text = temp.replace("__ENTITY__", ent)
    s = text.index(ent)
    e = s + len(ent)
    return {
        "id": f"utt_{random.randint(1000,9999)}",
        "text": text,
        "entities": [{"start": s, "end": e, "label": lab}]
    }

def write(path, n):
    with open(path,"w") as f:
        for _ in range(n):
            json.dump(build(), f); f.write("\n")

if __name__=="__main__":
    os.makedirs("data", exist_ok=True)
    write("data/train.jsonl", 1200)
    write("data/dev.jsonl",   300)
