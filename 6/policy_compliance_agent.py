def evaluate_request(request):
    violations = []
    review_notes = []

    request_type = request["request_type"].strip().lower()
    approval = request["approval"].strip().lower()
    contains_pii = request["contains_pii"].strip().lower()
    external_sharing = request["external_sharing"].strip().lower()

    try:
        amount = float(request["amount"])
    except ValueError:
        amount = 0

    if request_type == "expense":
        if approval != "yes":
            violations.append("Expense requires approval.")
        if amount > 1000 and approval != "yes":
            violations.append("Expense above 1000 requires approval.")

    elif request_type == "employee_data_export":
        if approval != "yes":
            violations.append("Employee data export requires approval.")

    elif request_type == "client_data_share":
        if contains_pii == "yes" and external_sharing == "yes":
            if approval != "yes":
                violations.append("External PII sharing requires approval.")
            else:
                review_notes.append("External PII sharing detected; manual privacy review is required.")

    elif request_type == "software_install":
        if approval != "yes":
            violations.append("Software installation requires approval.")

    else:
        violations.append("Unknown request type. Manual policy review is required.")

    if violations:
        return "NON-COMPLIANT", "BLOCK", violations, review_notes

    if review_notes:
        return "CONDITIONAL", "MANUAL REVIEW", [], review_notes

    return "COMPLIANT", "ALLOW", [], []


def get_input():
    print("\nPOLICY COMPLIANCE AGENT")
    print("=" * 45)
    print("Enter the details of your request.")
    print("Request types: expense, employee_data_export,")
    print("client_data_share, software_install")
    print()

    request = {}
    request["employee"] = input("Employee name: ")
    request["department"] = input("Department: ")
    request["request_type"] = input("Request type: ")
    request["amount"] = input("Amount (enter 0 if not applicable): ")
    request["contains_pii"] = input("Does it contain personal data/PII? (yes/no): ")
    request["external_sharing"] = input("Will it be shared externally? (yes/no): ")
    request["approval"] = input("Has required approval been obtained? (yes/no): ")
    return request


def main():
    print("Policy Compliance Agent")
    print("Type 'exit' when asked for the employee name to stop.")

    while True:
        request = get_input()

        if request["employee"].strip().lower() == "exit":
            print("\nAgent stopped.")
            break

        status, action, violations, review_notes = evaluate_request(request)

        print("\n" + "-" * 45)
        print("COMPLIANCE RESULT")
        print("-" * 45)
        print(f"Employee : {request['employee']}")
        print(f"Status   : {status}")
        print(f"Action   : {action}")

        if violations:
            print("\nReason:")
            for item in violations:
                print(f"- {item}")

        if review_notes:
            print("\nReview:")
            for item in review_notes:
                print(f"- {item}")

        if not violations and not review_notes:
            print("\nReason:")
            print("- All applicable policy rules were satisfied.")

        print("-" * 45)

        again = input("\nEvaluate another request? (yes/no): ")
        if again.strip().lower() != "yes":
            print("\nAgent stopped.")
            break


if __name__ == "__main__":
    main()
