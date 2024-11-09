import csv
import tqdm
import ollama

categories = {
    "bank_statement": """Bank 1 of Testing
Customer Support: 1-800-555-1234
www.fakebankdomain.com
Account Holder: John Doe 1
Account Number: XXXX-XXXX-XXXX-6781
Statement Period: 2023-01
Date Description Debit ($) Credit ($)
01/01/2023 Direct Deposit 326.73
20/01/2023 Direct Deposit 424.61
04/01/2023 ACH Payment 215.70
03/01/2023 Check Deposit 464.84
20/01/2023 Debit Card Purchase 695.38
04/01/2023 POS Purchase 136.10
27/01/2023 Online Transfer 285.34
05/01/2023 ACH Payment 436.96
16/01/2023 POS Purchase 203.32
15/01/2023 POS Purchase 613.07
28/01/2023 POS Purchase 189.97
07/01/2023 Debit Card Purchase 65.51
08/01/2023 Check Deposit 592.13
20/01/2023 Debit Card Purchase 289.50
28/01/2023 POS Purchase 10.44
24/01/2023 Loan Repayment 541.03
Bank 1 - Confidential Statement | Page 1 Bank 1 of Testing
Customer Support: 1-800-555-1234
www.fakebankdomain.com
End of Statement
Bank 1 - Confidential Statement | Page 2""",
    "invoice": """Invoice Date :
15 December 2023
Invoice to :
MORGAN MAXWELL
Magazine Design $50
Proposal Design $70
Brochure Design $30
Letterhead Design $20
ITEM DESC RIPTION PRIC E
$1 70SUBTOTAL :
SEND PAYMENT TO
Bank No:
Bank Name:
123-456-7890
Studio Shodwe
CONTACT
hello@reallygreatsite.com
+123-456-7890
T IMMERMA N
INDUS T RIES
 INVOICE
""",
    "drivers_licence": """DRIVING LICENSE

1. TROTTER

2.DEL

drivers license no. p99999999 expires 00-00-00

joe a sample
123 any street
anytown, any state 99999
')‘ sex: m hair: black
ht: 6-03 wt: 200

eyes: brown dob: 01-01-81 =

3.30.8.58 ENGLAND
4.13-95 7-3wheeler WLA

8.3 WHEELER ONLY
""",
}

for category, example in categories.items():
    for i in tqdm.tqdm(range(10), desc=category):
        response = ollama.generate(
            model="llama3",
            system="""You generate variations of an example text. You must change the formatting. You must change the length / number of pages. You must use novel words not in the original text, but relevant. You are given a category from which you must generate. Do not output anything except the new text. The variation of the text should have the same general feeling but seem like it was from a different origin. It should start with <OUTPUT>.""",
            prompt=f"Category: {category}\nExample: <OUTPUT>\n{example}",
            options={"temperature": 1.0, "top_k": 90},
        )

        content = response["response"]
        result = content.split("<OUTPUT>")[-1]
        print(result)

        with open("synthetic_data.csv", "a+", newline="") as csvfile:
            writer = csv.writer(
                csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            writer.writerow([result, category])
