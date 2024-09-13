import React, { ReactNode } from 'react';
import {
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
} from '@mui/material';

interface CommonTableProps {
    columns: string[];
    data: Record<string, ReactNode>[];
    containerStyle?: React.CSSProperties; // Additional style for the container
    cellStyle?: React.CSSProperties;

}

const CommonTable: React.FC<CommonTableProps> = ({ columns, data, containerStyle, cellStyle }) => {
    return (
        <TableContainer component={Paper} style={containerStyle} >
            <Table >
                <TableHead>
                    <TableRow>
                        {columns.map((column, index) => (
                            <TableCell key={index} style={cellStyle} className='columns' >{column}</TableCell>
                        ))}
                    </TableRow>
                </TableHead>
                <TableBody>
                    {data.map((row, rowIndex) => (
                        <TableRow key={rowIndex}>
                            {columns.map((column, colIndex) => (
                                <TableCell key={colIndex} style={cellStyle}>{row[column]}</TableCell>
                            ))}
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    );
};

export default CommonTable;
