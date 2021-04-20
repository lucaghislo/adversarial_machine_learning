def generate_adversarials(batch_size):
    while True:
        x = []
        y = []

        for batch in range(batch_size):
            N = random.randint(0, 100)
            label = y_train[N]
            image = x_train[N]
            perturbations = adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)), label).numpy()
            epsilon = 0.1
            adversarial = image + perturbations * epsilon
            x.append(adversarial)
            y.append(y_train[N])
        
        x = np.asarray(x).reshape((batch_size, img_rows, img_cols, channels))
        y = np.asarray(y)
        
        yield x, y

x_adversarial_test, y_adversarial_test = next(generate_adversarials(10000))
print("Base accuracy on adversarial images:", model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))
x_adversarial_train, y_adversarial_train = next(generate_adversarials(20000))

model.fit(x_adversarial_train, y_adversarial_train,
          batch_size=32,
          epochs=10,
          validation_data=(x_test, y_test))

print("Defended accuracy on adversarial images:", model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))
print("Defended accuracy on regular images:", model.evaluate(x=x_test, y=y_test, verbose=0))