import requests

API_URL = "http://localhost:8000/ask"


def main():
    question = input("Enter your question: ")

    response = requests.post(API_URL, json={"question": question})

    if response.status_code == 200:
        data = response.json()
        print("\nANSWER:\n", data.get("answer"))
    else:
        print("❌ Error:", response.text)


if __name__ == "__main__":
    main()