import React, { Componennt } from 'react';
import {
    TouchableOpacity,
    Text,
    StyleSheet,
    View,
} from 'react-native';
import PropTypes from 'prop-types';

const CustomButton = ({ title, onPress}) => {
    return (
        <TouchableOpacity
            onPress={() => {onPress(); console.log('press')}}
            onPressIn={() => console.log('in')}
            onPressOut={() => console.log('out')}
            onLongPress={() => console.log('long')}
            >
            <View style={{backgroundColor:'white', borderWidth:1 ,borderColor:'gray', width:180, height:50, marginLeft:10, marginTop:25, borderRadius:10}}>
                <Text style= {{fontSize:25, textAlign:'center', marginVertical:10}}>{title}</Text>
            </View>

        </TouchableOpacity>
    )
};

CustomButton.defaultProps = {
    title: 'default',
    onPress: () => alert('default'),
};
CustomButton.propTypes = {
    title: PropTypes.string,
    onPress: PropTypes.func,
};

export default CustomButton;