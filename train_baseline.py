"""Train the baseline CNN on clean MNIST and report clean test metrics."""
from common import (evaluate, inference_time, load_mnist, make_model, save_model, save_results, set_seed,
                    train)

if __name__ == "__main__":
    set_seed()
    x_train, y_train = load_mnist(train=True)
    x_test, y_test = load_mnist(train=False)
    model = train(make_model(), x_train, y_train, epochs=5)
    save_model(model, "baseline")
    res = evaluate(model, x_test, y_test)
    res["timing"] = inference_time(model, x_test)
    save_results("baseline_clean", res)
    print(f"baseline clean: acc {res['accuracy']:.4f} loss {res['loss']:.4f} f1 {res['macro_f1']:.4f}")
