import React from 'react';
import styled from 'styled-components/native';
import Button from '../components/Button';
import {useSafeAreaInsets} from 'react-native-safe-area-context';
import Category_dict from '../others/Category_dict';

const Container = styled.ScrollView`
    /* align-items: center; */
    flex:1;
    
    /* background-color: #ffffff; */
    padding-top: 10px;
    padding-bottom: 100px;
    padding-right: ${({insets: {right}}) => right}px;
    padding-left: ${({insets: {left}}) => left}px;
    align-self: center;
    margin-bottom: 100px;
`;

const StyledText = styled.Text`
    font-size: 30px;
    margin: 10px;
    `;

const Categories = ({ navigation }) => {
    const insets = useSafeAreaInsets();
    console.log(insets);
    var loop = [];

    for (let i=0, l = Object.keys(Category_dict).length; i < l ; i++){
        loop.push(
            <Button title={Category_dict[Object.keys(Category_dict)[i]]} onPress={() => navigation.navigate('Detail',Object.keys(Category_dict)[i])} />
        );
    }
    return (
        // Object.keys(Category_dict)
        <Container insets={insets}>
            {loop}
            {/* <Button title={Category_dict[Object.keys(Category_dict)[{loop}]]} onPress={() => navigation.navigate('Detail')} /> */}
            {/* <Button title={Category_dict[Object.keys(Category_dict)[1]]} onPress={() => navigation.navigate('Detail')} /> */}
        </Container>
    );
};

export default Categories;