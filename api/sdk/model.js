const tf = require('@tensorflow/tfjs-node');

function normalized(data){ // x1 & x2 & x3
    x1 = (data[0] - 42.6618) / 10.5959
    x2 = (data[1] - 88.7123) / 18.98
    x3 = (data[2] - 143.0393) / 23.0779
    return [x1, x2, x3]
}

function denormalized(data){
    y1 = (data[0] * 9.1734) + 74.77191
    y2 = (data[1] * 14.7892) + 49.84045
    y3 = (data[2] * 24.014) + 159.6989
    return [y1, y2, y3]
}


async function predict(data){
    let in_dim = 3;
    
    data = normalized(data);
    shape = [1, in_dim];

    tf_data = tf.tensor2d(data, shape);

    try{
        // path load in public access => github
        const path = 'https://raw.githubusercontent.com/zakkyakhmed/zakkyakhmed-UAS_SC_20_ZAKKY-AKHMAD-ZAKARIYA_41419010008/main/public/ex_model/model.json?token=5094739943:AAGfQPWdyjbiL5xLMfnXjO8OgqaPlGf6HRk';
        const model = await tf.loadGraphModel(path);
        
        predict = model.predict(
                tf_data
        );
        result = predict.dataSync();
        return denormalized( result );
        
    }catch(e){
      console.log(e);
    }
}

module.exports = {
    predict: predict 
}
