import React from 'react';
import Navigation from './navigations';
import { LogBox } from 'react-native';

export default function App() {
    LogBox.ignoreAllLogs();
    return <Navigation />
}