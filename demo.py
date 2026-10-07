"""Integrated router demo. No real actions or external customer data."""
import argparse, json
from experiment import ROOT, load_model, classify
from rag import Retriever, answer


def main():
    p = argparse.ArgumentParser();
    p.add_argument('question');
    p.add_argument('--model', default='gemma3:1b');
    args = p.parse_args()
    report_path = ROOT / 'results/tuned.json'
    if not report_path.exists(): raise SystemExit('Run training and evaluation first.')
    threshold = json.loads(report_path.read_text())['threshold']
    tok, model = load_model(adapter=True);
    model.eval()
    label, margin = classify(tok, model, args.question)
    print(json.dumps({'intent': label, 'uncalibrated_margin': margin, 'threshold': threshold}))
    if margin < threshold:
        print('Please clarify: are you asking about an existing order, a return, a refund, or a general store policy?')
    elif label in ['SHIPPING_POLICY', 'RETURN_POLICY']:
        docs = Retriever().search(args.question)
        print(answer(args.question, docs, args.model)['text'])
    else:
        print(
            f'DEMO ONLY: would route to {label}. A real system must authenticate the customer, validate the order state and confirm any change. No action was performed.')


if __name__ == '__main__': main()
