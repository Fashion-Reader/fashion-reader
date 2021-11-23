import React from 'react';
import { TouchableOpacity } from 'react-native';
import styled from 'styled-components/native';

const Container = styled.View`
    background-color: #F4F0ED;
    padding: 10px;
    margin: 10px;
    width: 350;
    height: 80;
    border-radius: 15;
    align-items: center;
    justify-content: center;
`;

const Title = styled.Text`
    font-size: 24px;
    color: #000000;
`;

const Button = ({ title, onPress }) => {
    return (
        <TouchableOpacity onPress={onPress}>
            <Container>
                <Title>{title}</Title>
            </Container>
        </TouchableOpacity>
    );
};

export default Button;
