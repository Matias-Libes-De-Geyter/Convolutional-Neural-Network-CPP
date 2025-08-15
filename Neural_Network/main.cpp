#include <SFML/Graphics.hpp>
#include "NN/CNN.hpp"
#include "Classifier/TrainerClassifier.hpp"
#include "Classifier/Scope.hpp"
#include "Dataset/Dataset.hpp"

hyperparameters hyper = {
    output_dim : 10,
    hidden_layer_sizes : {256, 128},
    learning_rate : 0.001,
    dropout_rate : 0.2,
    max_epochs : 50,
    n_train_samples : 10000,
    mini_batch_size : 32,
    n_val_samples : 1000,

    early_stopping : true,
    patience : 10,

    filters : { 32, 64 },
    kernel_size : 3,
    padding : 0,
    stride : 2,
    img_size : 28,
    database : "numbers"
};

int main() {
    CNN model(hyper);

    bool learning = false;
    print("Train ? (y/n)"); char a; std::cin >> a;
    if (a == 'y') learning = true;

    bool store = true;

    if (learning) {

        std::vector<Matrix> dummy_input = { Matrix(hyper.img_size, hyper.img_size) };
        model.forward(dummy_input, true);

        Scope scope(model, hyper);

        TrainerClassifier trainer(model, hyper);

        Dataset train = DataLoader(hyper, "train");
        Dataset validation = DataLoader(hyper, "validation");

        trainer.set_scope(scope);
        trainer.set_data(train, validation);
        print("Data has been successfully imported");

        trainer.run(store);
        model.saveWeights("executable/model_weights.txt", "executable/model_kernels.txt");
        print("Weights & Kernels saved !");

    }
    else {

        model.loadWeights("executable/model_weights.txt", "executable/model_kernels.txt");
        print("Weights & Kernels loaded !");

    }

    // Window init
    sf::RenderWindow window(sf::VideoMode({ 800, 800 }), "Deep Learning with Adam Optimizer");
    window.setFramerateLimit(100);
    const int img_size = hyper.img_size;
    sf::View view({ img_size / 2, img_size / 2 }, { img_size + 2, img_size + 2 });
    window.setView(view);

    // Canvas init
    sf::RenderTexture canvas({ img_size, img_size });
    canvas.clear(sf::Color::White);
    sf::Sprite sprite(canvas.getTexture());

    // Cursor init
    sf::RectangleShape cursor({ 2, 2 });
    cursor.setFillColor(sf::Color(255, 255, 255, 0));
    cursor.setOutlineThickness(0.5);
    cursor.setOrigin({ cursor.getSize().x / 2, cursor.getSize().y / 2 });
    cursor.setOutlineColor(sf::Color(0, 0, 0));
    // Brush border with 20% opacity
    const float brush_size = 0.75;
    sf::CircleShape brush(brush_size * 2, 5);
    brush.setOrigin({ brush_size * 2, brush_size * 2 });
    brush.setFillColor(sf::Color(120, 0, 255, 50));
    // Brush center with 100% opacity
    sf::CircleShape brushCenter(brush_size, 5);
    brushCenter.setOrigin({ brush_size, brush_size });
    brushCenter.setFillColor(sf::Color(120, 0, 255, 255));

    // Main loop
    bool firstPress = true;
    while (window.isOpen()) {
        sf::Vector2f mousePos = window.mapPixelToCoords(sf::Mouse::getPosition(window));
        cursor.setPosition(mousePos);

        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();
            else if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
                // "Escape" closes the window
                if (keyPressed->scancode == sf::Keyboard::Scancode::Escape)
                    window.close();

                // "R" resets the canvas
                if (keyPressed->scancode == sf::Keyboard::Scancode::R) {
                    canvas.clear(sf::Color::White);
                    canvas.display();
                }

                // "A" gives number prediction from the canvas
                if (keyPressed->scancode == sf::Keyboard::Scancode::A) {
                    if (firstPress) {
                        d_vector pixels(img_size * img_size, 0);
                        for (int i = 0; i < img_size; i++)
                            for (int j = 0; j < img_size; j++)
                                pixels[j * img_size + i] = 1.f - static_cast<int>(canvas.getTexture().copyToImage().getPixel({ i, j }).g) / 255.f;

                        std::vector<Matrix> input_images;
                        input_images.push_back(flattenToMatrix(pixels, img_size, img_size));

                        model.forward(input_images, false);
                        print("The number you've drawn is ", model.getOutput().getMaxIndex(), " !!!");
                    }
                    firstPress = false;
                }
            }
            else if (const auto* keyPressed = event->getIf<sf::Event::KeyReleased>())
                if (keyPressed->scancode == sf::Keyboard::Scancode::A)
                    firstPress = true;
        }

        // If I left click, it draws
        if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
            if (mousePos.x + 1 < img_size && mousePos.x > 1) {
                if (mousePos.y + 1 < img_size && mousePos.y > 1) {
                    brushCenter.setPosition(mousePos);
                    brush.setPosition(mousePos);
                    canvas.draw(brushCenter);
                    canvas.draw(brush);
                    canvas.display();
                }
            }
        }

        // Updating each frame
        window.clear(sf::Color(64, 64, 64));
        window.draw(sprite);
        window.draw(cursor);
        window.display();
    }

    return 0;
}